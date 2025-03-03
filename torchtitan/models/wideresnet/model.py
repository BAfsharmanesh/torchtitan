from dataclasses import dataclass
from torchtitan.models.utils import weights_init
import torch.nn as nn


@dataclass
class ModelArgs:
    n_tot_layers: int = 50
    num_filters: int = 64
    width_factor: int = 2
    num_classes: int = 1000
    input_channels: int = 3
    input_size: int = 32

    def __post_init__(self):
        model_configs = {
            18: (ResNetBlock, [2, 2, 2, 2]),
            34: (ResNetBlock, [3, 4, 6, 3]),
            50: (BottleneckResNetBlock, [3, 4, 6, 3]),
            101: (BottleneckResNetBlock, [3, 4, 23, 3]),
            152: (BottleneckResNetBlock, [3, 8, 36, 3]),
        }
        assert self.n_tot_layers in model_configs, f"Unsupported ResNet configuration for {self.n_tot_layers} layers."
        self.layers = model_configs[self.n_tot_layers][1]
        self.block = model_configs[self.n_tot_layers][0]
        self.n_layers = sum(self.layers)+2


## Wide ResNet Model ##
""" The definition of wide-resnet.

Implemented from https://github.com/alpa-projects/alpa/blob/main/alpa/model/wide_resnet.py.
see also: https://arxiv.org/pdf/1605.07146.pdf

"""


# Defining AdaptiveNorm to work with 1x1 feature maps as well
class AdaptiveNorm(nn.Module):
    def __init__(self, num_features):
        super(AdaptiveNorm, self).__init__()
        self.batch_norm = nn.BatchNorm2d(num_features)
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, x):
        if x.size(2) == 1 and x.size(3) == 1:  # Spatial dimensions are 1x1
            # Use LayerNorm for 1x1 feature maps
            x = x.view(x.size(0), x.size(1))  # Flatten spatial dimensions
            x = self.layer_norm(x)
            x = x.view(x.size(0), x.size(1), 1, 1)  # Reshape back to original
            return x
        else:
            return self.batch_norm(x)


class ResNetBlock(nn.Module):
    """ResNet block."""

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = AdaptiveNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = AdaptiveNorm(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class BottleneckResNetBlock(nn.Module):
    """Bottleneck ResNet block."""

    def __init__(
        self, in_channels, out_channels, stride=1, downsample=None, width_factor=4
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, bias=False
        )
        self.bn1 = AdaptiveNorm(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels * width_factor,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = AdaptiveNorm(out_channels * width_factor)
        self.conv3 = nn.Conv2d(
            out_channels * width_factor,
            out_channels * 4,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.bn3 = AdaptiveNorm(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample_conv = downsample[0] if downsample is not None else None
        self.downsample_bn = downsample[1] if downsample is not None else None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample_bn is not None:
            residual = self.downsample_bn(self.downsample_conv(x))

        out += residual
        out = self.relu(out)
        return out

# class MySequential(nn.Sequential):
#     def __init__(self, *args):
#         super(MySequential, self).__init__(*args)

#     def forward(self, x):
#         for layer in self:
#             if layer is not None:
#                 x = layer(x)
#         return x


class WideResNet(nn.Module):
    """ResNet Model."""

    def __init__(self, model_args: ModelArgs):
        super().__init__()
                
        self.in_channels = model_args.num_filters
        conv1 = nn.Conv2d(
            3, model_args.num_filters, kernel_size=7, stride=2, padding=3, bias=False
        )
        bn1 = AdaptiveNorm(model_args.num_filters)
        relu = nn.ReLU(inplace=True)
        maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        layer0 = nn.Sequential(conv1, bn1, relu, maxpool)
        layer1 = self._make_layer(
            model_args.block, model_args.num_filters, model_args.layers[0], stride=1, width_factor=model_args.width_factor
        )
        layer2 = self._make_layer(
            model_args.block, model_args.num_filters * 2, model_args.layers[1], stride=2, width_factor=model_args.width_factor
        )
        layer3 = self._make_layer(
            model_args.block, model_args.num_filters * 4, model_args.layers[2], stride=2, width_factor=model_args.width_factor
        )
        layer4 = self._make_layer(
            model_args.block, model_args.num_filters * 8, model_args.layers[3], stride=2, width_factor=model_args.width_factor
        )
        avgpool = nn.AdaptiveAvgPool2d((1, 1))
        fc = nn.Linear(
            model_args.num_filters * 8 * (4 if model_args.block == BottleneckResNetBlock else 1), model_args.num_classes
        )
        layer5 = nn.Sequential(avgpool, nn.Flatten(1), fc)
        
        layer_list = [layer0, layer1, layer2, layer3, layer4, layer5]
        self.layers = nn.ModuleDict()
        layer_id = 0
        for i, layer in enumerate(layer_list):
            if isinstance(layer, nn.ModuleDict):
                for j, sub_layer in layer.items():
                    self.layers[f"{layer_id}"] = sub_layer
                    layer_id += 1
            else:
                self.layers[str(layer_id)] = layer
                layer_id += 1
        
        # self.layers = nn.Sequential(layer0, layer1, layer2, layer3, layer4, layer5)

    def _make_layer(self, block, out_channels, blocks, stride, width_factor):
        layers = []
        downsample = None
        size_tmp = out_channels * (4 if block == BottleneckResNetBlock else 1)
        if stride != 1 or self.in_channels != size_tmp:
            downsample = (
                nn.Conv2d(
                    self.in_channels,
                    size_tmp,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                AdaptiveNorm(size_tmp),
            )

        layers.append(
            block(self.in_channels, out_channels, stride, downsample, width_factor)
        )
        self.in_channels = size_tmp
        for _ in range(1, blocks):
            layers.append(
                block(self.in_channels, out_channels, width_factor=width_factor)
            )

        # return nn.Sequential(*layers)
        seq_layers = nn.ModuleDict()
        for layer_id, layer in enumerate(layers):
            seq_layers[str(layer_id)] = layer        
        
        return seq_layers

    def init_weights(self):
        """Initializes the weights of the model."""
        for m in self.modules():
            m.apply(weights_init)

    def forward(self, x):

        # forward pass through the layers
        for layer in self.layers.values():
            if layer is not None:
                x = layer(x)
        return x

    @classmethod
    def from_model_args(cls, model_args: ModelArgs) -> "WideResNet":
        """
        Initialize an instance of the model from a ModelArgs object.

        Args:
            model_args (ModelArgs): Model configuration arguments.

        Returns:
            ResNet: ResNet model.

        """
        return cls(model_args) 