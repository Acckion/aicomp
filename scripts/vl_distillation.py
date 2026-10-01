"""Training-only, GT-matched region feature/semantic distillation for D-FINE.

This is a bounded local hypothesis, not a reproduction of DK-DETR. Native
Hungarian matching, detection logits, boxes, DN and all D-FINE losses remain.
Teacher agreement gates feature supervision; teacher predictions never replace
GT labels. Validation/test never receive cached embeddings.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T

import train_baseline as baseline
from src.core import register
from src.data.dataset.coco_dataset import CocoDetection
from src.zoo.dfine.dfine import DFINE
from src.zoo.dfine.dfine_criterion import DFINECriterion


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@register()
class VLTrainCoco(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, teacher_cache,
                 return_masks=False, remap_mscoco_category=False):
        super().__init__(img_folder, ann_file, transforms, return_masks, remap_mscoco_category)
        cache = Path(teacher_cache)
        identity = json.loads((cache/'identity.json').read_text())
        assert Path(ann_file).name == 'train1600.json'
        assert file_sha(ann_file) == identity['train_annotations_sha256']
        assert (cache/'COMPLETE').exists()
        with np.load(cache/'targets.npz') as values:
            self.teacher_features = torch.from_numpy(values['features'].copy())
            self.teacher_valid = torch.from_numpy(values['valid'].copy())
            self.teacher_labels = torch.from_numpy(values['labels'].copy())
            self.annotation_to_row = {int(a):i for i,a in enumerate(values['annotation_ids'])}
            assert set(values['image_ids'].tolist()) <= set(self.ids)

    def load_item(self, index):
        image, target = super().load_item(index)
        # Match upstream's crowd, clamp and degenerate-box filtering exactly.
        annotations = [a for a in self.coco.imgToAnns[self.ids[index]] if not a.get('iscrowd',0)]
        ids=[]
        for a in annotations:
            x,y,w,h=a['bbox'];x0=max(0,min(image.width,x));y0=max(0,min(image.height,y))
            x1=max(0,min(image.width,x+w));y1=max(0,min(image.height,y+h))
            if x1>x0 and y1>y0:ids.append(a['id'])
        rows=torch.tensor([self.annotation_to_row[a] for a in ids],dtype=torch.int64)
        assert torch.equal(self.teacher_labels[rows],target['labels'])
        assert len(rows)==len(target['boxes'])
        target['vl_features']=self.teacher_features[rows].clone()
        target['vl_valid']=self.teacher_valid[rows].clone()
        target['vl_annotation_ids']=torch.tensor(ids,dtype=torch.int64)
        return image,target


def annotation_tensors(inputs):
    target=inputs[1]
    return tuple(target[k] for k in ('labels','area','iscrowd','vl_features','vl_valid','vl_annotation_ids') if k in target)


@register()
class VLSanitizeBoxes(T.SanitizeBoundingBoxes):
    def __init__(self,min_size=1):
        super().__init__(min_size=min_size,labels_getter=annotation_tensors)


@register()
class VLDFINE(DFINE):
    def __init__(self,backbone,encoder,decoder,feature_dim=768):
        super().__init__(backbone,encoder,decoder)
        dim=decoder.dec_score_head[-1].in_features
        self.vl_projector=nn.Sequential(nn.Linear(dim,512),nn.GELU(),nn.Linear(512,feature_dim))
        self._vl_features=None
        decoder.dec_score_head[-1].register_forward_pre_hook(self._capture_queries)

    def _capture_queries(self,module,inputs):
        if self.training:self._vl_features=inputs[0]

    def forward(self,x,targets=None):
        self._vl_features=None
        outputs=super().forward(x,targets)
        if self.training:
            assert self._vl_features is not None
            features=self._vl_features
            if 'dn_meta' in outputs:
                features=features[:,outputs['dn_meta']['dn_num_split'][0]:]
            assert features.shape[:2]==outputs['pred_logits'].shape[:2]
            outputs['vl_embeddings']=self.vl_projector(features)
        self._vl_features=None
        return outputs


@register()
class VLDFINECriterion(DFINECriterion):
    def __init__(self,matcher,weight_dict,losses,teacher_cache,alpha=0.2,gamma=2.0,
                 num_classes=80,reg_max=32,boxes_weight_format=None,share_matched_indices=False,
                 feature_weight=0.1,semantic_weight=0.02,ramp_epochs=2,temperature=0.1):
        super().__init__(matcher,weight_dict,losses,alpha,gamma,num_classes,reg_max,
                         boxes_weight_format,share_matched_indices)
        with np.load(Path(teacher_cache)/'targets.npz') as data:
            self.register_buffer('text_prototypes',torch.from_numpy(data['prototypes'].copy()))
        assert self.text_prototypes.shape[0]==num_classes
        self.feature_weight=feature_weight;self.semantic_weight=semantic_weight
        self.ramp_epochs=ramp_epochs;self.temperature=temperature
        self.last_teacher_matches=0;self.last_vl_gradient_norm=None
        self._gradient_reported=False

    def capture_gradient(self,gradient):
        if not torch.isfinite(gradient).all():
            raise RuntimeError('Non-finite semantic distillation gradient')
        self.last_vl_gradient_norm=float(gradient.float().norm())
        if self.last_vl_gradient_norm>0 and not self._gradient_reported:
            print(f'VL_GRADIENT_VALIDATED teacher_matches={self.last_teacher_matches} '
                  f'projected_query_gradient_norm={self.last_vl_gradient_norm:.6g}',flush=True)
            self._gradient_reported=True

    def distillation_losses(self,outputs,targets,indices,epoch):
        embeddings=outputs['vl_embeddings'].float()
        students=[];teachers=[];labels=[]
        for b,(src,dst) in enumerate(indices):
            # Hungarian returns CPU indices; reliability masks live beside
            # the CUDA targets during real training.
            src=src.to(embeddings.device);dst=dst.to(embeddings.device)
            valid=targets[b]['vl_valid'][dst]
            students.append(embeddings[b,src[valid]])
            teachers.append(targets[b]['vl_features'][dst[valid]].float())
            labels.append(targets[b]['labels'][dst[valid]])
        student=torch.cat(students);teacher=torch.cat(teachers);label=torch.cat(labels)
        self.last_teacher_matches=len(student)
        zero=embeddings.sum()*0
        if not len(student):return {'loss_vl_feature':zero,'loss_vl_semantic':zero}
        student=F.normalize(student,dim=-1)
        ramp=min(1.,(epoch+1)/max(1,self.ramp_epochs))
        result={'loss_vl_feature':self.feature_weight*ramp*(1-F.cosine_similarity(student,teacher,dim=-1)).mean(),
                'loss_vl_semantic':self.semantic_weight*ramp*F.cross_entropy(student@self.text_prototypes.float().T/self.temperature,label)}
        if embeddings.requires_grad and self.feature_weight+self.semantic_weight>0:
            embeddings.register_hook(self.capture_gradient)
        return result

    def forward(self,outputs,targets,**kwargs):
        result=super().forward(outputs,targets,**kwargs)
        if self.training and 'vl_embeddings' in outputs:
            indices=self.matcher({'pred_logits':outputs['pred_logits'],'pred_boxes':outputs['pred_boxes']},targets)['indices']
            result.update(self.distillation_losses(outputs,targets,indices,kwargs.get('epoch',0)))
        return result


# Keep the upstream runner untouched; allow only the explicitly new projector.
_original_load=baseline.BaselineSolver.load_tuning_state
def load_tuning_state(self,path):
    if not isinstance(self.model,VLDFINE):return _original_load(self,path)
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    weights=dict(checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model'])
    current=self.model.state_dict()
    for key in ('decoder.anchors','decoder.valid_mask'):
        if key in current:weights[key]=current[key]
    matched,info=self._matched_state(current,weights)
    assert not [k for k in info['missed'] if not k.startswith('vl_projector.')],info
    assert not info['unmatched'],info
    self.model.load_state_dict(matched,strict=False)
    print('VL_INIT native detector loaded exactly; new projector:',info,flush=True)
baseline.BaselineSolver.load_tuning_state=load_tuning_state

_original_save=baseline.atomic_save
def atomic_save(state,path):
    if Path(path).name.startswith('weights_epoch_'):
        state=dict(state)
        state['model']={k:v for k,v in state['model'].items() if not k.startswith('vl_projector.')}
        state['training_method']='GT-matched gated CLIP region distillation'
        state['note']='Training-only projector stripped; use native D-FINE inference config.'
    return _original_save(state,path)
baseline.atomic_save=atomic_save
