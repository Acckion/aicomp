"""Check official MAL formula/gradients and four-view geometry before launch."""
import ast
import json
from pathlib import Path
import random
import sys
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import tv_tensors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'D-FINE'))
from src.core import YAMLConfig
from mal_dfine_criterion import MALDFINECriterion, box_iou, box_cxcywh_to_xyxy
from dense_o2o_views import DenseO2OViews, _patch


def main():
    random.seed(20260929)
    torch.manual_seed(20260929)
    path = ROOT / 'experiments/model_sources/DEIMv2/engine/deim/deim_criterion.py'
    tree = ast.parse(path.read_text())
    method = next(n for cls in tree.body if isinstance(cls, ast.ClassDef)
                  for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'loss_labels_mal')
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {'torch': torch, 'F': F, 'box_iou': box_iou,
                 'box_cxcywh_to_xyxy': box_cxcywh_to_xyxy}
    exec(compile(module, str(path), 'exec'), namespace)
    criterion = MALDFINECriterion(None, {'loss_vfl': 1}, ['vfl'], gamma=1.5)
    for empty in (False, True):
        labels = torch.empty(0, dtype=torch.long) if empty else torch.tensor([2, 5])
        boxes = torch.empty((0, 4)) if empty else torch.tensor([[.5,.5,.2,.2],[.3,.3,.1,.1]])
        indices = [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))] if empty else [(torch.tensor([0, 2]), torch.tensor([0, 1]))]
        logits = torch.randn(1, 6, 12, requires_grad=True)
        outputs = {'pred_logits': logits, 'pred_boxes': torch.rand(1, 6, 4)}
        targets = [{'labels': labels, 'boxes': boxes}]
        for values in (None, torch.empty(0) if empty else torch.tensor([.1, .9])):
            ours = criterion.loss_labels_vfl(outputs, targets, indices, max(1,len(labels)), values)['loss_vfl']
            reference = namespace['loss_labels_mal'](criterion, outputs, targets, indices, max(1,len(labels)), values)['loss_mal']
            torch.testing.assert_close(ours, reference, rtol=0, atol=0)
            gradient = torch.autograd.grad(ours, logits, retain_graph=True)[0]
            assert torch.isfinite(gradient).all()
    class Dummy:
        def __len__(self): return 1
        def load_item(self, index):
            return Image.new('RGB', (800,800)), {'boxes': tv_tensors.BoundingBoxes(torch.tensor([[390.,390.,410.,410.]]), format='XYXY', canvas_size=(800,800)), 'labels':torch.tensor([7]), 'iscrowd':torch.tensor([0]), 'area':torch.tensor([400.]), 'image_id':torch.tensor([1]), 'orig_size':torch.tensor([800,800])}
    dummy = Dummy()
    image, target = dummy.load_item(0)
    result_image, result, _ = DenseO2OViews(p=1)(image,target,dummy)
    assert result_image.size == (800,800) and len(result['boxes']) == 4
    torch.testing.assert_close(result['area'], torch.full((4,),400.))
    assert result['labels'].tolist() == [7]*4
    empty = dict(target, boxes=tv_tensors.BoundingBoxes(torch.empty(0,4),format='XYXY',canvas_size=(800,800)), labels=torch.empty(0,dtype=torch.long),iscrowd=torch.empty(0,dtype=torch.long))
    patch_image, patch_boxes, patch_fields = _patch(image, empty,800)
    assert patch_image.size == (400,400) and patch_boxes.shape == (0,4)
    # Check boundary clipping recomputes area and preserves field alignment.
    edge = dict(target,boxes=tv_tensors.BoundingBoxes(torch.tensor([[0.,0.,500.,500.]]),format='XYXY',canvas_size=(800,800)))
    _, edge_boxes, edge_fields = _patch(image,edge,800)
    assert (edge_boxes >= 0).all() and (edge_boxes <= 400).all() and len(edge_boxes)==len(edge_fields['labels'])
    cfg = YAMLConfig(str(ROOT/'configs/dense_o2o800.yml'))
    dataset = cfg.train_dataloader.dataset
    view = DenseO2OViews(p=1)
    old_counts, new_counts = [], []
    original_short, mosaic_short = [], []
    for i in range(64):
        image, target = dataset.load_item(i)
        result_image,result,_ = view(image,target,dataset)
        old_counts.append(len(target['boxes']));new_counts.append(len(result['boxes']))
        height, width = image.size[1], image.size[0]
        original_wh = (target['boxes'][:,2:]-target['boxes'][:,:2]) * torch.tensor([800/width,800/height])
        original_short.extend(original_wh.min(1).values.tolist())
        mosaic_short.extend((result['boxes'][:,2:]-result['boxes'][:,:2]).min(1).values.tolist())
        assert len(result['boxes']) == len(result['labels']) == len(result['area']) == len(result['iscrowd'])
        assert (result['boxes'] >= 0).all() and (result['boxes'] <= 800).all()
        assert ((result['boxes'][:,2:] - result['boxes'][:,:2]) >=1).all()
    report = {'mal_formula_exact_official_equivalence':True,'normal_and_empty_gradients_finite':True,
              'four_patch_pixel_size_preserved':True,'sample_images':64,
              'original_mean_targets':sum(old_counts)/64,'mosaic_mean_targets':sum(new_counts)/64,
              'max_mosaic_targets':max(new_counts),'mosaic_probability':.25,
              'original_shortside_percentiles_10_25_50_75':torch.quantile(torch.tensor(original_short),torch.tensor([.1,.25,.5,.75])).tolist(),
              'mosaic_shortside_percentiles_10_25_50_75':torch.quantile(torch.tensor(mosaic_short),torch.tensor([.1,.25,.5,.75])).tolist(),
              'minimum_visible_box_fraction':.3,
              'note':'Target-centered local-patch Mosaic only; not full DEIM recipe.'}
    dest=ROOT/'experiments/mechanisms/dense_correctness.json';dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__ == '__main__':main()
