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


import torch


def try_gpu(e):
    if torch.cuda.is_available():
        return e.cuda()
    return e


def im2col(im: torch.Tensor, kw: int, kh: int, stride=1, pad=0):
    N, C, H, W = im.shape
    out_h = (H + 2 * pad - kh) // stride + 1
    out_w = (W + 2 * pad - kw) // stride + 1
    p2d = (pad, pad, pad, pad)
    im = torch.nn.functional.pad(
        im,
        p2d,
        "constant",
        0,
    )
    col = torch.zeros(N, C, kw, kh, out_h, out_w)

    for y in range(kh):
        y_max = y + stride * out_h
        for x in range(kw):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = im[:, :, y:y_max:stride, x:x_max:stride]

    col = col.permute(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)

    return im
