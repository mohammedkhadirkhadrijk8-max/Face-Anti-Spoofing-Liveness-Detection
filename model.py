"""
model.py  –  Hybrid Face Anti-Spoofing Network
================================================
Architecture overview
---------------------
Three parallel feature branches are fused before a classification head:

1. **Deep Branch** (MobileNetV3-Small pretrained on ImageNet)
   – Input: RGB tensor (3, 128, 128)
   – Output: 576-d feature vector

2. **Texture Branch** (lightweight CNN)
   – Input: LBP map (1, 128, 128)
   – Output: 64-d feature vector

3. **Colour Branch** (lightweight CNN)
   – Input: YCrCb channels (3, 128, 128)
   – Output: 64-d feature vector

Fusion → FC layers → Binary output (0 = Spoof, 1 = Real)
"""

import torch
import torch.nn as nn
import torchvision.models as models


# ---------------------------------------------------------------------------
#  Small helper CNN used for LBP & YCrCb branches
# ---------------------------------------------------------------------------

class _LightCNN(nn.Module):
    """Tiny 4-conv feature extractor that works on arbitrary # of input channels."""

    def __init__(self, in_channels: int, out_features: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),  # 128→64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),           # 64→32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),          # 32→16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, stride=2, padding=1),         # 16→8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),                              # 8→1
        )
        self.fc = nn.Linear(128, out_features)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ---------------------------------------------------------------------------
#  Main hybrid model
# ---------------------------------------------------------------------------

class FaceAntiSpoofNet(nn.Module):
    """
    Hybrid anti-spoofing classifier that fuses deep, texture, and colour cues.

    Parameters
    ----------
    backbone : str
        ``'mobilenet_v3_small'`` (default) or ``'resnet18'``.
    num_classes : int
        2 for binary (spoof vs real).
    dropout : float
        Dropout probability before the final FC layer.
    pretrained : bool
        Whether to load ImageNet-pretrained weights for the backbone.
    """

    def __init__(self, backbone: str = "mobilenet_v3_small",
                 num_classes: int = 2, dropout: float = 0.4,
                 pretrained: bool = True):
        super().__init__()

        # ── Deep branch (pretrained backbone) ──
        if backbone == "mobilenet_v3_small":
            base = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            )
            deep_out = base.classifier[0].in_features        # 576
            base.classifier = nn.Identity()                   # strip head
            # MobileNetV3 expects 224×224 but works fine at 128 – output pool is adaptive
            self.deep_branch = base
        elif backbone == "resnet18":
            base = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if pretrained else None
            )
            deep_out = base.fc.in_features                    # 512
            base.fc = nn.Identity()
            self.deep_branch = base
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # ── Texture branch (LBP) ──
        self.texture_branch = _LightCNN(in_channels=1, out_features=64)

        # ── Colour branch (YCrCb) ──
        self.colour_branch = _LightCNN(in_channels=3, out_features=64)

        # ── Fusion + classifier ──
        fused_dim = deep_out + 64 + 64          # 576 + 64 + 64 = 704
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),

            nn.Linear(128, num_classes),
        )

        self._init_custom_weights()

    def _init_custom_weights(self):
        """Kaiming-initialise the non-pretrained parts of the network."""
        for module in [self.texture_branch, self.colour_branch, self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                            nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, rgb, ycrcb, lbp):
        """
        Parameters
        ----------
        rgb   : (B, 3, 128, 128) – normalised RGB
        ycrcb : (B, 3, 128, 128) – YCrCb channels
        lbp   : (B, 1, 128, 128) – LBP texture map

        Returns
        -------
        logits : (B, 2)
        """
        feat_deep    = self.deep_branch(rgb)          # (B, 576)
        feat_texture = self.texture_branch(lbp)       # (B, 64)
        feat_colour  = self.colour_branch(ycrcb)      # (B, 64)

        fused = torch.cat([feat_deep, feat_texture, feat_colour], dim=1)
        logits = self.classifier(fused)
        return logits


# ---------------------------------------------------------------------------
#  Quick standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = FaceAntiSpoofNet(pretrained=False)
    print(model)
    B = 4
    rgb   = torch.randn(B, 3, 128, 128)
    ycrcb = torch.randn(B, 3, 128, 128)
    lbp   = torch.randn(B, 1, 128, 128)
    out   = model(rgb, ycrcb, lbp)
    print("Output shape:", out.shape)          # (4, 2)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
