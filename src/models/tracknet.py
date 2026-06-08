"""Arquitectura TrackNet para deteccion de la pelota.

Red con encoder VGG16 preentrenado en ImageNet adaptado a 9 canales de entrada
(3 frames RGB consecutivos apilados) y decoder tipo U-Net con skip connections.
La salida es un unico canal sigmoide del mismo tamano que la entrada: un heatmap
donde la pelota aparece como un pico.

Codigo portado tal cual de la celda de arquitectura de
``old_notebooks/02_ball_tracking.ipynb``. La logica de congelado de capas y de
adaptacion de la primera conv se conserva intacta para poder cargar el
checkpoint entrenado (``tracknet_best.pth``) sin desajustes de pesos.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models


class TrackNetV2(nn.Module):
    """TrackNet con encoder VGG16 preentrenado adaptado a 9 canales de entrada."""

    def __init__(self, in_channels=9, out_channels=1):
        super().__init__()

        vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        feats = list(vgg.features.children())

        # Adaptar la primera conv: 3 -> in_channels promediando los pesos
        # preentrenados triplicados, para preservar el conocimiento de ImageNet.
        orig_conv = feats[0]
        new_first = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        with torch.no_grad():
            new_first.weight.data = orig_conv.weight.data.repeat(1, 3, 1, 1) / 3.0
            new_first.bias.data = orig_conv.bias.data.clone()
        feats[0] = new_first

        # Bloques del encoder. Las skip connections se toman antes de cada MaxPool.
        self.enc1 = nn.Sequential(*feats[0:4])    # 64ch,  H×W
        self.pool1 = feats[4]
        self.enc2 = nn.Sequential(*feats[5:9])    # 128ch, H/2
        self.pool2 = feats[9]
        self.enc3 = nn.Sequential(*feats[10:16])  # 256ch, H/4
        self.pool3 = feats[16]
        self.enc4 = nn.Sequential(*feats[17:23])  # 512ch, H/8
        self.pool4 = feats[23]
        self.bottleneck = nn.Sequential(*feats[24:30])  # 512ch, H/16 (se omite pool5)

        # Congelar todo el encoder salvo enc1[0] (la primera conv, que se adapta a 9ch).
        for name, param in self.enc1.named_parameters():
            if not name.startswith("0."):
                param.requires_grad = False
        for enc in [self.enc2, self.enc3, self.enc4, self.bottleneck]:
            for param in enc.parameters():
                param.requires_grad = False

        # Decoder: Upsample + 2×(Conv-BN-ReLU) con skip connections tipo U-Net.
        self.dec4 = self._conv_block(512 + 512, 512)
        self.dec3 = self._conv_block(512 + 256, 256)
        self.dec2 = self._conv_block(256 + 128, 128)
        self.dec1 = self._conv_block(128 + 64,  64)
        self.head = nn.Conv2d(64, out_channels, kernel_size=1)

    @staticmethod
    def _conv_block(in_ch, out_ch):
        """Bloque de dos convoluciones 3x3 con BatchNorm y ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool1(s1))
        s3 = self.enc3(self.pool2(s2))
        s4 = self.enc4(self.pool3(s3))
        b = self.bottleneck(self.pool4(s4))

        def up(t, ref):
            """Upsample bilineal de ``t`` al tamano espacial de ``ref``."""
            return F.interpolate(t, size=ref.shape[2:], mode="bilinear", align_corners=False)

        d4 = self.dec4(torch.cat([up(b,  s4), s4], dim=1))
        d3 = self.dec3(torch.cat([up(d4, s3), s3], dim=1))
        d2 = self.dec2(torch.cat([up(d3, s2), s2], dim=1))
        d1 = self.dec1(torch.cat([up(d2, s1), s1], dim=1))

        return torch.sigmoid(self.head(d1))
