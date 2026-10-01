"""Official infrared images with native RGB-coordinate detection annotations.

Reads paired files lazily from the official ZIP, without expanding the dataset.
Matching image dimensions are checked; they do not establish pixel registration.
"""
import io
import os
from pathlib import Path
import zipfile

from PIL import Image
import torch

from src.core import register
from src.data.dataset.coco_dataset import CocoDetection
from src.data._misc import convert_to_tv_tensor


@register()
class ZipIRCocoDetection(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, archive,
                 return_masks=False, remap_mscoco_category=False):
        super().__init__(img_folder,ann_file,transforms,return_masks,remap_mscoco_category)
        if remap_mscoco_category:
            raise ValueError('AICOMP class IDs must remain unchanged')
        self.archive_path=str(Path(archive).resolve())
        self._archive=None
        self._archive_pid=None
        self.members={}
        with zipfile.ZipFile(self.archive_path) as source:
            for member in source.namelist():
                path=Path(member)
                if len(path.parts)>=2 and path.parts[-2]=='infrared':
                    if path.name in self.members:raise ValueError('Duplicate IR image name')
                    self.members[path.name]=member
        for image in self.coco.dataset['images']:
            assert Path(image['file_name']).name in self.members, 'Missing official IR pair'

    def __getstate__(self):
        state=self.__dict__.copy()
        state['_archive']=None
        state['_archive_pid']=None
        return state

    def load_ir(self, metadata):
        pid=os.getpid()
        if self._archive is None or self._archive_pid!=pid:
            if self._archive is not None:self._archive.close()
            self._archive=zipfile.ZipFile(self.archive_path)
            self._archive_pid=pid
        member=self.members[Path(metadata['file_name']).name]
        with Image.open(io.BytesIO(self._archive.read(member))) as source:
            image=source.convert('RGB')
        assert image.size==(metadata['width'],metadata['height']), 'IR/RGB dimension mismatch'
        return image

    def load_item(self,idx):
        image_id=self.ids[idx]
        metadata=self.coco.loadImgs(image_id)[0]
        image=self.load_ir(metadata)
        target={'image_id':image_id,
                'image_path':f"{self.archive_path}:{self.members[Path(metadata['file_name']).name]}",
                'annotations':self.coco.imgToAnns.get(image_id,[])}
        image,target=self.prepare(image,target)
        target['idx']=torch.tensor([idx])
        target['boxes']=convert_to_tv_tensor(target['boxes'],key='boxes',spatial_size=image.size[::-1])
        if 'masks' in target:target['masks']=convert_to_tv_tensor(target['masks'],key='masks')
        return image,target
