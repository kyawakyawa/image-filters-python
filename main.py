"""
MIT License

Copyright (c) 2021 kyawakyawa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import argparse
import torch
import torchvision
from utils import try_gpu
from box_filter import box_filter

from PIL import Image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--image",
        action="store",
        nargs=1,
        const=None,
        default=None,
        type=str,
        choices=None,
        help="filepath to image",
        metavar=None,
    )

    args = parser.parse_args()

    im = Image.open(args.image[0])

    torch.set_grad_enabled(False)

    im = torchvision.transforms.functional.to_tensor(im)

    H, W = im.shape[1:3]

    im = try_gpu(im)

    im = im.unsqueeze(0)

    unfold = torch.nn.Unfold(9, padding=4, stride=1)
    im = unfold(im)
    im = box_filter(im, 3)

    im = im.reshape(im.shape[1], H, W)

    im = torchvision.transforms.functional.to_pil_image(im)

    im.save("output.jpg")
