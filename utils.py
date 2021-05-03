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
