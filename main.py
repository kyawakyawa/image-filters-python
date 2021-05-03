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
