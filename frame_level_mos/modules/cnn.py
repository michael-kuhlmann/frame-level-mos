import typing as tp

from padertorch.contrib.je.modules import conv
from padertorch.utils import to_list
from torch import Tensor, nn
from torch.nn.utils import parametrizations, parametrize


def apply_weight_norm(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        parametrizations.weight_norm(m)


class CNN1d(conv.CNN1d):
    sequence_last = True

    def forward(self, x: Tensor, *args, **kwargs):
        """
        Args:
            x (Tensor): Input of shape (batch, time, features).
        """
        if not self.sequence_last:
            x = x.transpose(1, 2)
        x, *outputs = super().forward(x, *args, **kwargs)
        if not self.sequence_last:
            x = x.transpose(1, 2)
        return x, *outputs


class CNNTranspose1d(conv.CNNTranspose1d):
    sequence_last = True

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                parametrize.remove_parametrizations(m, 'weight')
            except ValueError:  # This module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def forward(self, x: Tensor, *args, **kwargs):
        """
        Args:
            x (Tensor): Input of shape (batch, time, features).
        """
        if not self.sequence_last:
            x = x.transpose(1, 2)
        x, *outputs = super().forward(x, *args, **kwargs)
        if not self.sequence_last:
            x = x.transpose(1, 2)
        return x, *outputs


def build_cnn1d(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    num_layers: int,
    kernel_size: int,
    activation_fn: tp.Union[str, tp.Callable],
    stride: tp.Union[int, tp.List[int]] = 1,
    dilation: tp.Union[int, tp.List[int]] = 1,
    output_activation_fn: tp.Optional[tp.Union[str, tp.Callable]] = None,
    pre_activation: bool = True,
    norm: tp.Optional[str] = None,
    normalize_skip_convs: bool = False,
    block_size: tp.Optional[int] = 2,
    input_layer: bool = True,
    output_layer: bool = True,
    transpose: bool = False,
    use_weight_norm: bool = False,
    sequence_last: bool = True,
):
    out_channels = [hidden_channels] * (num_layers - 1) + [out_channels]
    dilations = [1 for _ in range(num_layers)]
    if not isinstance(stride, list):
        strides = [1 for _ in range(num_layers)]
        if not transpose:
            strides[-1] = stride
        else:
            strides[0] = stride
    else:
        stride = to_list(stride, num_layers)

    residual_connections = [None for _ in range(num_layers)]
    if block_size is not None:
        block_dilation = to_list(dilation, block_size)
        for i in range(
            int(input_layer), num_layers-int(output_layer), block_size
        ):
            residual_connections[i] = i+block_size
            dilations[i:i+block_size] = block_dilation
    if output_activation_fn is not None:
        if pre_activation and not norm:
            raise RuntimeError(
                "Output activation becomes uneffective if pre_activation=True "
                "and norm=None."
            )
        output_layer = False
        activation_fn = (
            [activation_fn for _ in range(num_layers)] + [output_activation_fn]
        )
        activation_fn[0] = "identity"

    factory = CNN1d if not transpose else CNNTranspose1d
    m = factory(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=strides,
        dilation=dilations,
        residual_connections=residual_connections,
        activation_fn=activation_fn,
        pre_activation=pre_activation,
        norm=norm,
        normalize_skip_convs=normalize_skip_convs,
        input_layer=input_layer,
        output_layer=output_layer,
    )
    m.sequence_last = sequence_last
    if use_weight_norm:
        m = m.apply(apply_weight_norm)
    return m
