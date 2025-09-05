import torch
import torch.nn as nn

# Placeholder that loads DeepFillV2 from OpenMMLab
class DeepFillV2(nn.Module):
    def __init__(self):
        super(DeepFillV2, self).__init__()
        from mmedit.models import build_model
        from mmcv import Config

        # Config from OpenMMLab
        cfg = Config.fromfile("configs/inpainting/deepfillv2/deepfillv2_places_256x256.py")
        self.model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    
    def forward(self, x):
        return self.model(x)
