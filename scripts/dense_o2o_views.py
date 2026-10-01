"""Pixel-preserving four-image Mosaic, a Dense O2O augmentation ablation.

DEIM uses Mosaic/Mixup to increase one-to-one positive supervision. Here four
target-centered local patches are assembled, rather than downscaling four full
images, to retain each object's baseline-800 pixel size. Three other sources are
loaded only from the supplied training dataset's raw load_item (never val/test).
This intentionally tests a limited Dense O2O variant, not the full DEIM recipe.
"""
import random
import torch
from torch import nn
from PIL import Image
from torchvision import tv_tensors
from src.core import register


def _patch(image, target, output_size, min_visible_fraction=0.3):
    size = output_size
    half = size // 2
    width, height = image.size
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    boxes = target['boxes'].as_subclass(torch.Tensor).clone()
    boxes *= boxes.new_tensor([size / width, size / height, size / width, size / height])
    original_area = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    if len(boxes):
        anchor = random.randrange(len(boxes))
        center = (boxes[anchor, :2] + boxes[anchor, 2:]) / 2
        left = round(max(0, min(size - half, float(center[0]) - half / 2 + random.uniform(-half/8, half/8))))
        top = round(max(0, min(size - half, float(center[1]) - half / 2 + random.uniform(-half/8, half/8))))
    else:
        left, top = random.randint(0, half), random.randint(0, half)
    boxes -= boxes.new_tensor([left, top, left, top])
    boxes.clamp_(0, half)
    visible_area = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    keep = (((boxes[:, 2] - boxes[:, 0]) >= 1) & ((boxes[:, 3] - boxes[:, 1]) >= 1)
            & (visible_area >= original_area * min_visible_fraction))
    return image.crop((left, top, left + half, top + half)), boxes[keep], {
        key: target[key][keep].clone() for key in ('labels', 'iscrowd') if key in target}


@register()
class DenseO2OViews(nn.Module):
    def __init__(self, p=0.25, output_size=800, min_visible_fraction=0.3):
        super().__init__()
        if not 0 <= p <= 1 or output_size % 2:
            raise ValueError('Probability must be in [0,1] and output size even')
        self.p, self.output_size = p, output_size
        if not 0 <= min_visible_fraction <= 1:
            raise ValueError('Visible fraction must be in [0,1]')
        self.min_visible_fraction = min_visible_fraction

    def forward(self, *inputs):
        image, target, dataset = inputs if len(inputs) > 1 else inputs[0]
        if random.random() >= self.p:
            return image, target, dataset
        if 'masks' in target or 'keypoints' in target:
            raise ValueError('This four-view Mosaic currently supports box labels only')
        sources = [(image, target)] + [dataset.load_item(random.randrange(len(dataset))) for _ in range(3)]
        half = self.output_size // 2
        canvas = Image.new('RGB', (self.output_size, self.output_size))
        fields = {'labels': [], 'iscrowd': []}
        all_boxes = []
        for slot, (source_image, source_target) in enumerate(sources):
            patch, boxes, annotations = _patch(source_image, source_target, self.output_size,
                                              self.min_visible_fraction)
            x, y = (slot % 2) * half, (slot // 2) * half
            canvas.paste(patch, (x, y))
            boxes += boxes.new_tensor([x, y, x, y])
            all_boxes.append(boxes)
            for key in fields:
                if key in annotations:
                    fields[key].append(annotations[key])
        boxes = torch.cat(all_boxes)
        result = dict(target)
        result['boxes'] = tv_tensors.BoundingBoxes(boxes, format='XYXY',
                                                  canvas_size=(self.output_size, self.output_size))
        for key, values in fields.items():
            if values:
                if len(values) != 4:
                    raise ValueError('Mixed annotations missing per-object field: ' + key)
                result[key] = torch.cat(values)
        result['area'] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        result['orig_size'] = torch.tensor([self.output_size, self.output_size])
        # Keep the primary image_id only as metadata; COCO evaluation never runs
        # on augmented training views. size follows the composite image.
        if 'size' in result:
            result['size'] = torch.tensor([self.output_size, self.output_size])
        return canvas, result, dataset
