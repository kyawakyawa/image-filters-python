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


def x_derivative_operator(input: torch.Tensor, padding=1):
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
        filters[:, kernel_size_sq * i + 3] = 0.5
        filters[:, kernel_size_sq * i + 5] = -0.5

    return (filters @ col).reshape(N, channel, H, W)


def y_derivative_operator(input: torch.Tensor, padding=1):
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
        filters[:, kernel_size_sq * i + 1] = 0.5
        filters[:, kernel_size_sq * i + 7] = -0.5

    return (filters @ col).reshape(N, channel, H, W)


def x_prewitt_filter(input: torch.Tensor, padding=1):
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
        filters[:, kernel_size_sq * i + 0] = 1.0 / 6.0
        filters[:, kernel_size_sq * i + 3] = 1.0 / 6.0
        filters[:, kernel_size_sq * i + 6] = 1.0 / 6.0
        filters[:, kernel_size_sq * i + 2] = -1.0 / 6.0
        filters[:, kernel_size_sq * i + 5] = -1.0 / 6.0
        filters[:, kernel_size_sq * i + 8] = -1.0 / 6.0

    return (filters @ col).reshape(N, channel, H, W)


def y_prewitt_filter(input: torch.Tensor, padding=1):
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
        filters[:, kernel_size_sq * i + 0] = 1.0 / 6.0
        filters[:, kernel_size_sq * i + 1] = 1.0 / 6.0
        filters[:, kernel_size_sq * i + 2] = 1.0 / 6.0
        filters[:, kernel_size_sq * i + 6] = -1.0 / 6.0
        filters[:, kernel_size_sq * i + 7] = -1.0 / 6.0
        filters[:, kernel_size_sq * i + 8] = -1.0 / 6.0

    return (filters @ col).reshape(N, channel, H, W)


def x_sobel_filter(input: torch.Tensor, padding=1):
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
        filters[:, kernel_size_sq * i + 0] = 1.0 / 8.0
        filters[:, kernel_size_sq * i + 3] = 2.0 / 8.0
        filters[:, kernel_size_sq * i + 6] = 1.0 / 8.0
        filters[:, kernel_size_sq * i + 2] = -1.0 / 8.0
        filters[:, kernel_size_sq * i + 5] = -2.0 / 8.0
        filters[:, kernel_size_sq * i + 8] = -1.0 / 8.0

    return (filters @ col).reshape(N, channel, H, W)


def y_sobel_filter(input: torch.Tensor, padding=1):
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
        filters[:, kernel_size_sq * i + 0] = 1.0 / 8.0
        filters[:, kernel_size_sq * i + 1] = 2.0 / 8.0
        filters[:, kernel_size_sq * i + 2] = 1.0 / 8.0
        filters[:, kernel_size_sq * i + 6] = -1.0 / 8.0
        filters[:, kernel_size_sq * i + 7] = -2.0 / 8.0
        filters[:, kernel_size_sq * i + 8] = -1.0 / 8.0

    return (filters @ col).reshape(N, channel, H, W)


def steerable_filter(input: torch.Tensor, theta: float, padding=1):
    kernel_size = 3
    N, channel, H, W = input.shape
    # im2col
    col = torch.nn.Unfold(kernel_size, padding=padding, stride=1)(input)

    if col.shape[1] % channel != 0:
        return None

    kernel_size_sq = col.shape[1] // channel
    # col.shape[1] == kernel_size_sq * channel

    filters_x = torch.zeros(
        channel, col.shape[1], device=input.device, dtype=input.dtype
    )
    filters_y = torch.zeros(
        channel, col.shape[1], device=input.device, dtype=input.dtype
    )

    for i in range(channel):
        cos_theta = torch.cos(
            torch.tensor([theta], device=input.device, dtype=torch.float32)
        )

        filters_x[:, kernel_size_sq * i + 0] = cos_theta * 1.0 / 8.0
        filters_x[:, kernel_size_sq * i + 3] = cos_theta * 2.0 / 8.0
        filters_x[:, kernel_size_sq * i + 6] = cos_theta * 1.0 / 8.0
        filters_x[:, kernel_size_sq * i + 2] = cos_theta * -1.0 / 8.0
        filters_x[:, kernel_size_sq * i + 5] = cos_theta * -2.0 / 8.0
        filters_x[:, kernel_size_sq * i + 8] = cos_theta * -1.0 / 8.0

        sin_theta = torch.sin(
            torch.tensor([theta], device=input.device, dtype=torch.float32)
        )

        filters_y[:, kernel_size_sq * i + 0] = sin_theta * 1.0 / 8.0
        filters_y[:, kernel_size_sq * i + 1] = sin_theta * 2.0 / 8.0
        filters_y[:, kernel_size_sq * i + 2] = sin_theta * 1.0 / 8.0
        filters_y[:, kernel_size_sq * i + 6] = sin_theta * -1.0 / 8.0
        filters_y[:, kernel_size_sq * i + 7] = sin_theta * -2.0 / 8.0
        filters_y[:, kernel_size_sq * i + 8] = sin_theta * -1.0 / 8.0

    filters = filters_x + filters_y

    return (filters @ col).reshape(N, channel, H, W)
