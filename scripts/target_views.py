"""Train-only mixture of baseline geometric augmentation and small-target views."""
import random
import torch
from torch import nn
from torchvision import tv_tensors
import torchvision.transforms.v2 as T
from src.core import register
from src.data.transforms._transforms import RandomIoUCrop


def crop_target(image,target,left,top,width,height):
    if 'masks' in target or 'keypoints' in target:
        raise ValueError('Target-aware views currently support box annotations only')
    boxes=target['boxes'].as_subclass(torch.Tensor).clone()
    boxes[:,[0,2]]-=left;boxes[:,[1,3]]-=top
    boxes[:,[0,2]]=boxes[:,[0,2]].clamp(0,width)
    boxes[:,[1,3]]=boxes[:,[1,3]].clamp(0,height)
    keep=(boxes[:,2]-boxes[:,0]>=1) & (boxes[:,3]-boxes[:,1]>=1)
    result=dict(target)
    result['boxes']=tv_tensors.BoundingBoxes(boxes[keep],format='XYXY',canvas_size=(height,width))
    for key in ('labels','iscrowd'):
        if key in target:result[key]=target[key][keep]
    result['area']=(boxes[keep,2]-boxes[keep,0])*(boxes[keep,3]-boxes[keep,1])
    result['orig_size']=torch.tensor([width,height])
    return image.crop((left,top,left+width,top+height)),result


@register()
class MixedTargetViews(nn.Module):
    def __init__(self,p=.35,min_fraction=.45,max_fraction=.7,max_target_area_fraction=.0025,zoom_p=.5,iou_p=.8):
        super().__init__()
        assert 0<=p<=1 and 0<min_fraction<=max_fraction<=1
        self.p=p;self.min_fraction=min_fraction;self.max_fraction=max_fraction
        self.max_target_area_fraction=max_target_area_fraction
        self.zoom=T.RandomZoomOut(fill=0,p=zoom_p)
        self.iou=RandomIoUCrop(p=iou_p)

    def forward(self,*inputs):
        sample=inputs if len(inputs)>1 else inputs[0]
        image,target,dataset=sample
        width,height=image.size
        boxes=target['boxes'].as_subclass(torch.Tensor)
        areas=(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
        candidates=torch.where((areas>0) & (areas/(width*height)<=self.max_target_area_fraction))[0]
        if len(candidates) and random.random()<self.p:
            anchor=boxes[random.choice(candidates.tolist())]
            fraction=random.uniform(self.min_fraction,self.max_fraction)
            cw,ch=max(2,round(width*fraction)),max(2,round(height*fraction))
            # Require the chosen object to be fully inside the crop.
            lx,hx=max(0,float(anchor[2])-cw),min(float(anchor[0]),width-cw)
            ly,hy=max(0,float(anchor[3])-ch),min(float(anchor[1]),height-ch)
            if lx<=hx and ly<=hy:
                left,top=round(random.uniform(lx,hx)),round(random.uniform(ly,hy))
                image,target=crop_target(image,target,left,top,cw,ch)
                return image,target,dataset
        # p=0 is the exact geometric-augmentation control.
        sample=self.zoom(sample)
        return self.iou(sample) if len(sample[1]['boxes']) else sample
