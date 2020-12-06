import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import AttentionConv, AttentionStem


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, groups=1, base_width=64):
        super(Bottleneck, self).__init__()
        self.stride = stride
        width = int(out_channels * (base_width / 64.)) * groups

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            AttentionConv(width, width, kernel_size=7, padding=3, groups=8),
            nn.BatchNorm2d(width),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(width, self.expansion * out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.expansion * out_channels),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.stride >= 2:
            out = F.avg_pool2d(out, (self.stride, self.stride))

        out += self.shortcut(x)
        out = F.relu(out)

        return out


class Model(nn.Module):
    def __init__(self, block, num_blocks, dataset, num_classes=1000, stem=False):
        super(Model, self).__init__()
        self.in_places = 64

        if dataset in ['CIFAR10', 'CIFAR100']:
            if stem:
                self.init = nn.Sequential(
                    # CIFAR10
                    AttentionStem(in_channels=3, out_channels=64, kernel_size=4, stride=1, padding=2, groups=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                )
            else:
                self.init = nn.Sequential(
                    # CIFAR10
                    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                )
        elif dataset == 'IMAGENET':
            if stem:
                self.init = nn.Sequential(
                    # For ImageNet
                    AttentionStem(in_channels=3, out_channels=64, kernel_size=4, stride=1, padding=2, groups=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(4, 4)
                )
            else:
                self.init = nn.Sequential(
                    # For ImageNet
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                )

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.dense = nn.Linear(512 * block.expansion, num_classes)
        self.num_blocks = num_blocks

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_places, planes, stride))
            self.in_places = planes * block.expansion
        return nn.Sequential(*layers)

    def get_attention(self, layer, aggregated=True, norm=True, unpad=True):
        """Get attention map of an attention layer.
        `norm` and `unpad` do not matter if `aggregated` is False

        Ref: https://discuss.pytorch.org/t/how-to-unfold-a-tensor-into-sliding-windows-and-then-fold-it-back/55890/7

        Args:
            layer: attention layer
            aggregated: whether to aggregate across footprints by folding
            norm: whether to normalize to eliminate cumulative effect
            unpad: whether to unpad the attention

        Return:
            attn: retrived attention map
        """
        assert hasattr(layer, 'attn_raw'), 'Layer does not have attention'
        assert layer.attn_raw is not None, 'Attention not computed yet'

        attn = layer.attn_raw
        if not aggregated:
            return attn

        # Prepare the dimension
        kernel_size = layer.kernel_size
        stride = layer.stride
        B, N, C, H, W, K = attn.shape
        padding = 2 * (kernel_size//2)

        # (B, N, C, H, W, K) -> (B*N*C, K, H*W)
        attn = attn.reshape([B*N*C, H*W, K]).permute(0, 2, 1)

        # Fold the attenion to aggregate
        attn = F.fold(attn, (H+padding, W+padding),
                      kernel_size=kernel_size, stride=stride)

        # (B*N*C, 1, H_pad, W_pad) -> (B, N*C, H_pad, W_pad)
        attn = attn.reshape([B, N*C, H+padding, W+padding])

        # Normalize the cummulative effect of sliding
        if norm:
            norm_mask = F.conv2d(torch.ones(1, 1, H, H),
                                 torch.ones(1, 1, kernel_size, kernel_size),
                                 padding=padding).to(attn.device)
            attn = attn / norm_mask

        # Unpad the attention
        if unpad:
            attn = attn[:, :, padding//2:padding//2+H, padding//2:padding//2+W]
        return attn

    def get_all_attention(self, aggregated=True, norm=True, unpad=True):
        """Get attention of all layers

        Return:
            attn_dict: dictionary of attention with the structure:
                attn_dict[l_name][m_name], where l_name is layer name and
                m_name is module name
        """
        attn_dict = {}
        for l in range(len(self.num_blocks)):
            l_name = 'layer{}'.format(l+1)
            attn_dict[l_name] = {}
            layer = getattr(self, l_name)
            for m in range(self.num_blocks[l]):
                attn = self.get_attention(
                    layer[m].conv2[0], aggregated, norm, unpad)
                attn_dict[l_name][str(m)] = attn
        return attn_dict

    def forward(self, x):
        out = self.init(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.dense(out)

        # attn_dict = self.get_all_attention()
        return out


def ResNet26(num_classes=1000, stem=False, dataset='CIFAR10'):
    return Model(Bottleneck, [1, 2, 4, 1], num_classes=num_classes, stem=stem)


def ResNet38(num_classes=1000, stem=False, dataset='CIFAR10'):
    return Model(Bottleneck, [2, 3, 5, 2], num_classes=num_classes, stem=stem)


def ResNet50(num_classes=1000, stem=False, dataset='CIFAR10'):
    return Model(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, stem=stem)


def get_model_parameters(model):
    total_parameters = 0
    for layer in list(model.parameters()):
        layer_parameter = 1
        for l in list(layer.size()):
            layer_parameter *= l
        total_parameters += layer_parameter
    return total_parameters


# temp = torch.randn((2, 3, 224, 224))
# model = ResNet38(num_classes=1000, stem=True)
# print(get_model_parameters(model))
