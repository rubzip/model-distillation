import torch
import torch.nn as nn


def _make_conv_block(in_c, out_c, use_depthwise):
    if use_depthwise and in_c > 1:
        return nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, groups=in_c),
            nn.Conv2d(in_c, out_c, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


class VGGishStudent(nn.Module):
    def __init__(
        self,
        channels: list[int],
        fc_sizes: list[int],
        use_depthwise: bool = False,
        in_channels: int = 1,
        input_shape=(96, 64),
    ):
        super().__init__()

        blocks = []
        in_c = in_channels
        for c in channels:
            blocks.append(_make_conv_block(in_c, c, use_depthwise))
            in_c = c

        self.features = nn.Sequential(*blocks)

        # Calculate flattened size
        dummy_input = torch.zeros(1, in_channels, *input_shape)
        with torch.no_grad():
            dummy_out = self.features(dummy_input)
            flattened_size = dummy_out.view(1, -1).size(1)

        self.flatten = nn.Flatten()

        fc_layers = []
        in_fc = flattened_size
        for fc in fc_sizes:
            fc_layers.append(nn.Linear(in_fc, fc))
            fc_layers.append(nn.ReLU(inplace=True))
            in_fc = fc

        # The last layer in standard VGGish FCs does NOT have ReLU if it's the embedding layer
        # But wait, original VGGish usually has ReLU at the end of FCs? Actually, the embedding itself might be pre-activation or post.
        # We will keep it as nn.Linear for the final embedding layer if it's the last one, or add ReLU and then nothing?
        # Typically the final embedding is just a Linear projection (no ReLU). Let's remove the last ReLU.
        fc_layers.pop()

        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, x):
        # x is expected to be [B, C, F, T] or [B, F, T]
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


class VGGishFactory:
    @staticmethod
    def create_vggish200k(input_shape=(128, 101)):
        # VGGish200k: Conv1(8), Conv2(16), Conv3(32), Conv4(64), Conv5(64)
        # FC1(256), FC3(128)
        return VGGishStudent(
            channels=[8, 16, 32, 64, 64],
            fc_sizes=[256, 128],
            use_depthwise=False,
            input_shape=input_shape,
        )

    @staticmethod
    def create_vggish200k_mobile(input_shape=(128, 101), embed_dim: int = 32):
        # Mobile version uses depthwise separable convs.
        # In the production model (Table 24), FC1 is 256 and FC2 is 32 (Linear).
        return VGGishStudent(
            channels=[8, 16, 32, 64, 64],
            fc_sizes=[256, embed_dim],
            use_depthwise=True,
            input_shape=input_shape,
        )

    @staticmethod
    def create_vggish20k(input_shape=(128, 101)):
        # VGGish20k: Conv1(4), Conv2(8), Conv3(16), Conv4(16), Conv5(32)
        # FC1(64), FC3(128)
        return VGGishStudent(
            channels=[4, 8, 16, 16, 32],
            fc_sizes=[64, 128],
            use_depthwise=False,
            input_shape=input_shape,
        )
