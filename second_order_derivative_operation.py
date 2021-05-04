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


def x_2nd_derivative_operator(input: torch.Tensor, padding=1):
    kernel_size = 3
    N, channel, H, W = input.shape
    # im2col
    col = torch.nn.Unfold(kernel_size, padding=padding, stride=1)(input)

    if col.shape[1] % channel != 0:
        return None

    kernel_size_sq = col.shape[1] // channel
    # col.shape[1] == kernel_size_sq * channel

    filters = torch.zeros(
        channel, col.shape[1], device=input.device, dtype=input.dtype
    )

    for i in range(channel):
        filters[:, kernel_size_sq * i + 1] = 1.0
        filters[:, kernel_size_sq * i + 4] = -2.0
        filters[:, kernel_size_sq * i + 7] = 1.0

    return (filters @ col).reshape(N, channel, H, W)


def y_2nd_derivative_operator(input: torch.Tensor, padding=1):
    kernel_size = 3
    N, channel, H, W = input.shape
    # im2col
    col = torch.nn.Unfold(kernel_size, padding=padding, stride=1)(input)

    if col.shape[1] % channel != 0:
        return None

    kernel_size_sq = col.shape[1] // channel
    # col.shape[1] == kernel_size_sq * channel

    filters = torch.zeros(
        channel, col.shape[1], device=input.device, dtype=input.dtype
    )

    for i in range(channel):
        filters[:, kernel_size_sq * i + 3] = 1.0
        filters[:, kernel_size_sq * i + 4] = -2.0
        filters[:, kernel_size_sq * i + 5] = 1.0

    return (filters @ col).reshape(N, channel, H, W)


def laplacian_filter(input: torch.Tensor, padding=1):
    kernel_size = 3
    N, channel, H, W = input.shape
    # im2col
    col = torch.nn.Unfold(kernel_size, padding=padding, stride=1)(input)

    if col.shape[1] % channel != 0:
        return None

    kernel_size_sq = col.shape[1] // channel
    # col.shape[1] == kernel_size_sq * channel

    filters = torch.zeros(
        channel, col.shape[1], device=input.device, dtype=input.dtype
    )

    for i in range(channel):
        filters[:, kernel_size_sq * i + 1] = 1.0
        filters[:, kernel_size_sq * i + 3] = 1.0
        filters[:, kernel_size_sq * i + 4] = -4.0
        filters[:, kernel_size_sq * i + 5] = 1.0
        filters[:, kernel_size_sq * i + 7] = 1.0

    return (filters @ col).reshape(N, channel, H, W)
