from typing import Any, Union, BinaryIO, Optional
import pathlib
import torch
from torchvision.utils import _log_api_usage_once, make_grid
from PIL import Image


def save_image(
        tensor: torch.tensor,
        fp: Union[str, pathlib.Path, BinaryIO],
        format: Optional[str] = None,
        gray_image: bool = False,
        **kwargs
):
    """
    通过convert('L')可以在保存灰度图时减少存储开销
    torchvision.utils原来的实现是将单通道数据直接拼接三次, 因此出来后做一个灰度图的转换
    Args:
        gray_image: 输入的数据是否是灰度图像数据
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(tensor)
    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    if gray_image:
        im = im.convert('L')
    im.save(fp, format=format)
