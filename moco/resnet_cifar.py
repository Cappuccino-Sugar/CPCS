"""
This code is based on the Torchvision repository, which was licensed under the BSD 3-Clause.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2, 2], in_channel=3, zero_init_residual=False, num_classes=128,
                 feature_only=False, test=True, **kwargs):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.input_resolution = 32

        # self.attnpool = AttentionPool2d(self.input_resolution // 32, 512,
        #                                 8, 512)
        # self.FPN = FPN()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        self.feature_only = feature_only
        if not feature_only:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

            if test:
                self.fc = nn.Linear(512, num_classes)
            else:
                self.fc = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, num_classes))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # if not self.feature_only:
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out

    # def forward(self, x):
    #     out = F.relu(self.bn1(self.conv1(x)))
    #     out = self.layer1(out)
    #     x2 = self.layer2(out)
    #     x3 = self.layer3(x2)
    #     x4 = self.layer4(x3)
    #     x4 = self.attnpool(x4)
    #     x = self.FPN((x2, x3, x4))
    #
    #     # # if not self.feature_only:
    #     # out = self.avgpool(out)
    #     # out = torch.flatten(out, 1)
    #     return x

# def get_resnet(name, **kwargs):
#     resnet18 = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], **kwargs)
#     resnet34 = ResNet(block=BasicBlock, layers=[3, 4, 6, 3], **kwargs)
#     resnet50 = ResNet(block=Bottleneck, layers=[3, 4, 6, 3], **kwargs)
#
#     resnets = {
#         "ResNet18": resnet18,
#         "ResNet34": resnet34,
#         "ResNet50": resnet50,
#     }
#     if name not in resnets.keys():
#         raise KeyError(f"{name} is not a valid ResNet version")
#     return resnets[name]

def resnet18_cifar(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet34_cifar(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def resnet50_cifar(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


# class AttentionPool2d(nn.Module):
#     def __init__(self,
#                  spacial_dim: int,
#                  embed_dim: int,
#                  num_heads: int,
#                  output_dim: int = None):
#         super().__init__()
#         self.spacial_dim = spacial_dim
#         self.positional_embedding = nn.Parameter(
#             torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5)
#         self.k_proj = nn.Linear(embed_dim, embed_dim)
#         self.q_proj = nn.Linear(embed_dim, embed_dim)
#         self.v_proj = nn.Linear(embed_dim, embed_dim)
#         self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
#         self.num_heads = num_heads
#         # residual
#         self.connect = nn.Sequential(
#             nn.Conv2d(embed_dim, output_dim, 1, stride=1, bias=False),
#             nn.BatchNorm2d(output_dim))
#
#     def resize_pos_embed(self, pos_embed, input_shpae):
#         """Resize pos_embed weights.
#         Resize pos_embed using bicubic interpolate method.
#         Args:
#             pos_embed (torch.Tensor): Position embedding weights.
#             input_shpae (tuple): Tuple for (downsampled input image height,
#                 downsampled input image width).
#             pos_shape (tuple): The resolution of downsampled origin training
#                 image.
#             mode (str): Algorithm used for upsampling:
#                 ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
#                 ``'trilinear'``. Default: ``'nearest'``
#         Return:
#             torch.Tensor: The resized pos_embed of shape [B, C, L_new]
#         """
#         assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
#         pos_h = pos_w = self.spacial_dim
#         cls_token_weight = pos_embed[:, 0]
#         pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]
#         pos_embed_weight = pos_embed_weight.reshape(
#             1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
#         pos_embed_weight = F.interpolate(pos_embed_weight,
#                                          size=input_shpae,
#                                          align_corners=False,
#                                          mode='bicubic')
#         cls_token_weight = cls_token_weight.unsqueeze(1)
#         pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
#         # pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
#         return pos_embed_weight.transpose(-2, -1)
#
#     def forward(self, x):
#         B, C, H, W = x.size()
#         res = self.connect(x)
#         x = x.reshape(B, C, -1)  # NC(HW)
#         # x = torch.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(1+HW)
#         pos_embed = self.positional_embedding.unsqueeze(0)
#         pos_embed = self.resize_pos_embed(pos_embed, (H, W))  # NC(HW)
#         x = x + pos_embed.to(x.dtype)  # NC(HW)
#         x = x.permute(2, 0, 1)  # (HW)NC
#         x, _ = F.multi_head_attention_forward(
#             query=x,
#             key=x,
#             value=x,
#             embed_dim_to_check=x.shape[-1],
#             num_heads=self.num_heads,
#             q_proj_weight=self.q_proj.weight,
#             k_proj_weight=self.k_proj.weight,
#             v_proj_weight=self.v_proj.weight,
#             in_proj_weight=None,
#             in_proj_bias=torch.cat(
#                 [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
#             bias_k=None,
#             bias_v=None,
#             add_zero_attn=False,
#             dropout_p=0,
#             out_proj_weight=self.c_proj.weight,
#             out_proj_bias=self.c_proj.bias,
#             use_separate_proj_weight=True,
#             training=self.training,
#             need_weights=False)
#         x = x.permute(1, 2, 0).reshape(B, -1, H, W)
#         x = x + res
#         x = F.relu(x, True)
#
#         return x

# def linear_layer(in_dim, out_dim, bias=False):
#     return nn.Sequential(nn.Linear(in_dim, out_dim, bias),
#                          nn.BatchNorm1d(out_dim), nn.ReLU(True))
#
# def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
#     return nn.Sequential(
#         nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
#         nn.BatchNorm2d(out_dim), nn.ReLU(True))
#
# class CoordConv(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size=3,
#                  padding=1,
#                  stride=1):
#         super().__init__()
#         self.conv1 = conv_layer(in_channels + 2, out_channels, kernel_size,
#                                 padding, stride)

# class FPN(nn.Module):
#     def __init__(self,
#                  in_channels=[512, 512, 1024],
#                  out_channels=[256, 512, 1024]):
#         super(FPN, self).__init__()
#         # text projection
#         self.txt_proj = linear_layer(in_channels[2], out_channels[2])
#         # fusion 1: v5 & seq -> f_5: b, 1024, 13, 13
#         self.f1_v_proj = conv_layer(in_channels[2], out_channels[2], 1, 0)
#         self.norm_layer = nn.Sequential(nn.BatchNorm2d(out_channels[2]),
#                                         nn.ReLU(True))
#         # fusion 2: v4 & fm -> f_4: b, 512, 26, 26
#         self.f2_v_proj = conv_layer(in_channels[1], out_channels[1], 3, 1)
#         self.f2_cat = conv_layer(out_channels[2] + out_channels[1],
#                                  out_channels[1], 1, 0)
#         # fusion 3: v3 & fm_mid -> f_3: b, 512, 52, 52
#         self.f3_v_proj = conv_layer(in_channels[0], out_channels[0], 3, 1)
#         self.f3_cat = conv_layer(out_channels[0] + out_channels[1],
#                                  out_channels[1], 1, 0)
#         # fusion 4: f_3 & f_4 & f_5 -> fq: b, 256, 26, 26
#         self.f4_proj5 = conv_layer(out_channels[2], out_channels[1], 3, 1)
#         self.f4_proj4 = conv_layer(out_channels[1], out_channels[1], 3, 1)
#         self.f4_proj3 = conv_layer(out_channels[1], out_channels[1], 3, 1)
#         # aggregation
#         self.aggr = conv_layer(3 * out_channels[1], out_channels[1], 1, 0)
#         self.coordconv = nn.Sequential(
#             CoordConv(out_channels[1], out_channels[1], 3, 1),
#             conv_layer(out_channels[1], out_channels[1], 3, 1))
#
#     def forward(self, imgs):
#         # v3, v4, v5: 256, 52, 52 / 512, 26, 26 / 1024, 13, 13
#         v3, v4, v5 = imgs
#         # fusion 1: b, 1024, 13, 13
#         # text projection: b, 1024 -> b, 1024
#         # state = self.txt_proj(state).unsqueeze(-1).unsqueeze(
#         #     -1)  # b, 1024, 1, 1
#         f5 = self.f1_v_proj(v5)
#         # f5 = self.norm_layer(f5 * state)
#         f5 = self.norm_layer(f5)
#         # fusion 2: b, 512, 26, 26
#         f4 = self.f2_v_proj(v4)
#         # 利用插值方法，对输入的张量数组进行上\下采样操作，换句话说就是科学合理地改变数组的尺寸大小，尽量保持数据完整
#         f5_ = F.interpolate(f5, scale_factor=2, mode='bilinear')
#         f4 = self.f2_cat(torch.cat([f4, f5_], dim=1))
#         # fusion 3: b, 256, 26, 26
#         f3 = self.f3_v_proj(v3)
#         f3 = F.avg_pool2d(f3, 2, 2)
#         f3 = self.f3_cat(torch.cat([f3, f4], dim=1))
#         # fusion 4: b, 512, 13, 13 / b, 512, 26, 26 / b, 512, 26, 26
#         fq5 = self.f4_proj5(f5)
#         fq4 = self.f4_proj4(f4)
#         fq3 = self.f4_proj3(f3)
#         # query
#         fq5 = F.interpolate(fq5, scale_factor=2, mode='bilinear')
#         fq = torch.cat([fq3, fq4, fq5], dim=1)
#         fq = self.aggr(fq)
#         fq = self.coordconv(fq)
#         # b, 512, 26, 26
#         return fq
