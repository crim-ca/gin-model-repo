import torch
import thelper.nn
import torch.nn as nn
import torch.nn.functional as F

from .common import ActivatedBatchNorm
from .encoder import create_encoder
from .decoder import create_decoder
from .tta import SegmentatorTTA


class EncoderDecoderNet(nn.Module, SegmentatorTTA):
    def __init__(self, num_classes, enc_type='resnet50', dec_type='unet_scse',
                 num_filters=16, pretrained=False):
        super().__init__()
        self.output_channels = num_classes
        self.enc_type = enc_type
        self.dec_type = dec_type
        self.num_classes= num_classes
        assert enc_type in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                            'resnext101_32x4d', 'resnext101_64x4d',
                            'se_resnet50', 'se_resnet101', 'se_resnet152',
                            'se_resnext50_32x4d', 'se_resnext101_32x4d', 'senet154']
        assert dec_type in ['unet_scse', 'unet_seibn', 'unet_oc']

        encoder = create_encoder(enc_type, pretrained)
        Decoder = create_decoder(dec_type)

        self.encoder1 = encoder[0]
        self.encoder2 = encoder[1]
        self.encoder3 = encoder[2]
        self.encoder4 = encoder[3]
        self.encoder5 = encoder[4]

        self.pool = nn.MaxPool2d(2, 2)
        self.center = Decoder(self.encoder5.out_channels, num_filters * 32 * 2, num_filters * 32)

        self.decoder5 = Decoder(self.encoder5.out_channels + num_filters * 32, num_filters * 32 * 2,
                                num_filters * 16)
        self.decoder4 = Decoder(self.encoder4.out_channels + num_filters * 16, num_filters * 16 * 2,
                                num_filters * 8)
        self.decoder3 = Decoder(self.encoder3.out_channels + num_filters * 8, num_filters * 8 * 2, num_filters * 4)
        self.decoder2 = Decoder(self.encoder2.out_channels + num_filters * 4, num_filters * 4 * 2, num_filters * 2)
        self.decoder1 = Decoder(self.encoder1.out_channels + num_filters * 2, num_filters * 2 * 2, num_filters)

        self.logits = nn.Sequential(
            nn.Conv2d(num_filters * (16 + 8 + 4 + 2 + 1), 64, kernel_size=1, padding=0),
            ActivatedBatchNorm(64),
            nn.Conv2d(64, self.output_channels, kernel_size=1)
        )

    def forward(self, x):
        img_size = x.shape[2:]

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        c = self.center(self.pool(e5))
        e1_up = F.interpolate(e1, scale_factor=2, mode='bilinear', align_corners=False)

        d5 = self.decoder5(c, e5)
        d4 = self.decoder4(d5, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1_up)

        u5 = F.interpolate(d5, img_size, mode='bilinear', align_corners=False)
        u4 = F.interpolate(d4, img_size, mode='bilinear', align_corners=False)
        u3 = F.interpolate(d3, img_size, mode='bilinear', align_corners=False)
        u2 = F.interpolate(d2, img_size, mode='bilinear', align_corners=False)

        # Hyper column
        d = torch.cat((d1, u2, u3, u4, u5), 1)
        logits = self.logits(d)

        return logits

    def set_task(self, task):
        assert isinstance(task, thelper.tasks.Segmentation), "missing impl for non-segm task type"
        if self.final_block is None or self.num_classes != len(task.class_names):
            self.num_classes = len(task.class_names)
            self.final_block = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=self.mid_channels // 8,
                                out_channels=self.mid_channels // 16,
                                kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(in_channels=self.mid_channels // 16,
                                out_channels=self.num_classes,
                                kernel_size=1),
            )
        self.task = task

