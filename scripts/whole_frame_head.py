"""Full-frame pixel-detail adaptation with frozen visual feature extraction.

Both arms have identical 1088x1920 geometry. The sham arm reconstructs the
original canvas from the shared 800-square image, so it has no new pixels.
"""
import torch
from torch.nn import functional as F
import native_encoded_delta  # Register the shared original-image dataset.
from src.core import register
from src.zoo.dfine.dfine import DFINE


@register()
class WholeFrameHeadDFINE(DFINE):
    inference_input_protocol = 'whole_frame_head_rgb7_1088x1920_v1'

    def __init__(self, backbone, encoder, decoder, mode='real', size=800,
                 height=1088, width=1920, train_encoder=False):
        super().__init__(backbone,encoder,decoder)
        assert mode in ('real','sham')
        self.mode=mode;self.size=size;self.height=height;self.width=width
        self.train_encoder=bool(train_encoder)
        self.forced_size=None
        self.backbone.requires_grad_(False)
        self.encoder.requires_grad_(self.train_encoder)
        self.encoder.eval_spatial_size=None
        self.decoder.eval_spatial_size=None

    def prepare_inference_input(self,image,size):
        if size != self.size:
            raise ValueError('Nominal base size must be800; actual frame geometry is1088x1920')
        return native_encoded_delta.NativeEncodedDeltaDFINE.prepare_inference_input(self,image,size)

    def frame(self,packed):
        frames=[]
        for item in packed:
            h,w=map(int,item[6,0,:2].detach().cpu().tolist())
            assert h>0 and w>0
            image=item[None,3:6,:h,:w]
            if self.mode=='sham':
                image=F.interpolate(item[None,:3,:self.size,:self.size],size=(h,w),
                                    mode='bilinear',align_corners=False)
            frames.append(F.interpolate(image,size=(self.height,self.width),
                                         mode='bilinear',align_corners=False))
        return torch.cat(frames)

    def forward(self,packed,targets=None):
        assert packed.shape[1]==7
        self.backbone.eval();self.encoder.eval()
        with torch.no_grad():
            features=self.backbone(self.frame(packed))
        if self.train_encoder:
            features=self.encoder(features)
        else:
            with torch.no_grad():
                features=self.encoder(features)
        return self.decoder(features,targets)
