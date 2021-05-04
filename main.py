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
import math
import torch
import torchvision
from utils import try_gpu, to_gray_scale
from box_filter import box_filter
from shift_filter import shift_filter
from first_order_derivative_operation import (
    x_derivative_operator,
    y_derivative_operator,
    x_prewitt_filter,
    y_prewitt_filter,
    x_sobel_filter,
    y_sobel_filter,
    steerable_filter,
)
from second_order_derivative_operation import (
    x_2nd_derivative_operator,
    y_2nd_derivative_operator,
    laplacian_filter,
)

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

    # box filter
    im_box_filterd = box_filter(im, 5, 2)

    im_box_filterd = im_box_filterd.squeeze()

    im_box_filterd = torchvision.transforms.functional.to_pil_image(
        im_box_filterd
    )

    im_box_filterd.save("output-box-filtered.jpg")

    # shift filter
    im_shift_filterd = shift_filter(im, 5, 2)

    im_shift_filterd = im_shift_filterd.squeeze()

    im_shift_filterd = torchvision.transforms.functional.to_pil_image(
        im_shift_filterd
    )

    im_shift_filterd.save("output-shift-filtered.jpg")

    # x derivative operator
    im_x_derivative_operator = to_gray_scale(im)
    im_x_derivative_operator = x_derivative_operator(im_x_derivative_operator)
    im_x_derivative_operator = torch.abs(im_x_derivative_operator)
    mx = torch.max(im_x_derivative_operator)
    im_x_derivative_operator = im_x_derivative_operator / mx
    im_x_derivative_operator = im_x_derivative_operator.squeeze()
    im_x_derivative_operator = torchvision.transforms.functional.to_pil_image(
        im_x_derivative_operator
    )

    im_x_derivative_operator.save("output-x-derivative-operator.jpg")

    # y derivative operator
    im_y_derivative_operator = to_gray_scale(im)
    im_y_derivative_operator = y_derivative_operator(im_y_derivative_operator)
    im_y_derivative_operator = torch.abs(im_y_derivative_operator)
    mx = torch.max(im_y_derivative_operator)
    im_y_derivative_operator = im_y_derivative_operator / mx
    im_y_derivative_operator = im_y_derivative_operator.squeeze()
    im_y_derivative_operator = torchvision.transforms.functional.to_pil_image(
        im_y_derivative_operator
    )

    im_y_derivative_operator.save("output-y-derivative-operator.jpg")

    # x prewitt filter
    im_x_prewitt_filter = to_gray_scale(im)
    im_x_prewitt_filter = x_prewitt_filter(im_x_prewitt_filter)
    im_x_prewitt_filter = torch.abs(im_x_prewitt_filter)
    mx = torch.max(im_x_prewitt_filter)
    im_x_prewitt_filter = im_x_prewitt_filter / mx
    im_x_prewitt_filter = im_x_prewitt_filter.squeeze()
    im_x_prewitt_filter = torchvision.transforms.functional.to_pil_image(
        im_x_prewitt_filter
    )

    im_x_prewitt_filter.save("output-x-prewitt-filter.jpg")

    # y prewitt filter
    im_y_prewitt_filter = to_gray_scale(im)
    im_y_prewitt_filter = y_prewitt_filter(im_y_prewitt_filter)
    im_y_prewitt_filter = torch.abs(im_y_prewitt_filter)
    mx = torch.max(im_y_prewitt_filter)
    im_y_prewitt_filter = im_y_prewitt_filter / mx
    im_y_prewitt_filter = im_y_prewitt_filter.squeeze()
    im_y_prewitt_filter = torchvision.transforms.functional.to_pil_image(
        im_y_prewitt_filter
    )

    im_y_prewitt_filter.save("output-y-prewitt-filter.jpg")

    # x sobel filter
    im_x_sobel_filter = to_gray_scale(im)
    im_x_sobel_filter = x_sobel_filter(im_x_sobel_filter)
    im_x_sobel_filter = torch.abs(im_x_sobel_filter)
    mx = torch.max(im_x_sobel_filter)
    im_x_sobel_filter = im_x_sobel_filter / mx
    im_x_sobel_filter = im_x_sobel_filter.squeeze()
    im_x_sobel_filter = torchvision.transforms.functional.to_pil_image(
        im_x_sobel_filter
    )

    im_x_sobel_filter.save("output-x-sobel-filter.jpg")

    # y sobel filter
    im_y_sobel_filter = to_gray_scale(im)
    im_y_sobel_filter = y_sobel_filter(im_y_sobel_filter)
    im_y_sobel_filter = torch.abs(im_y_sobel_filter)
    mx = torch.max(im_y_sobel_filter)
    im_y_sobel_filter = im_y_sobel_filter / mx
    im_y_sobel_filter = im_y_sobel_filter.squeeze()
    im_y_sobel_filter = torchvision.transforms.functional.to_pil_image(
        im_y_sobel_filter
    )

    im_y_sobel_filter.save("output-y-sobel-filter.jpg")

    # steerable filter
    im_steerable_filter = to_gray_scale(im)
    im_steerable_filter = steerable_filter(
        im_steerable_filter, math.radians(45)
    )
    im_steerable_filter = torch.abs(im_steerable_filter)
    mx = torch.max(im_steerable_filter)
    im_steerable_filter = im_steerable_filter / mx
    im_steerable_filter = im_steerable_filter.squeeze()
    im_steerable_filter = torchvision.transforms.functional.to_pil_image(
        im_steerable_filter
    )

    im_steerable_filter.save("output-steerable_filter.jpg")

    # x 2nd derivative operator
    im_x_2nd_derivative_operator = to_gray_scale(im)
    im_x_2nd_derivative_operator = x_2nd_derivative_operator(
        im_x_2nd_derivative_operator
    )
    im_x_2nd_derivative_operator = torch.abs(im_x_2nd_derivative_operator)
    mx = torch.max(im_x_2nd_derivative_operator)
    im_x_2nd_derivative_operator = im_x_2nd_derivative_operator / mx
    im_x_2nd_derivative_operator = im_x_2nd_derivative_operator.squeeze()
    im_x_2nd_derivative_operator = (
        torchvision.transforms.functional.to_pil_image(
            im_x_2nd_derivative_operator
        )
    )

    im_x_2nd_derivative_operator.save("output-x-2nd-derivative-operator.jpg")

    # y 2nd derivative operator
    im_y_2nd_derivative_operator = to_gray_scale(im)
    im_y_2nd_derivative_operator = y_2nd_derivative_operator(
        im_y_2nd_derivative_operator
    )
    im_y_2nd_derivative_operator = torch.abs(im_y_2nd_derivative_operator)
    mx = torch.max(im_y_2nd_derivative_operator)
    im_y_2nd_derivative_operator = im_y_2nd_derivative_operator / mx
    im_y_2nd_derivative_operator = im_y_2nd_derivative_operator.squeeze()
    im_y_2nd_derivative_operator = (
        torchvision.transforms.functional.to_pil_image(
            im_y_2nd_derivative_operator
        )
    )

    im_y_2nd_derivative_operator.save("output-y-2nd-derivative-operator.jpg")

    # laplacian filter
    im_laplacian_filter = to_gray_scale(im)
    im_laplacian_filter = laplacian_filter(im_laplacian_filter)
    im_laplacian_filter = torch.abs(im_laplacian_filter)
    mx = torch.max(im_laplacian_filter)
    im_laplacian_filter = im_laplacian_filter / mx
    im_laplacian_filter = im_laplacian_filter.squeeze()
    im_laplacian_filter = torchvision.transforms.functional.to_pil_image(
        im_laplacian_filter
    )

    im_laplacian_filter.save("output-laplacian-filter.jpg")
