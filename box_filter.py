import torch


def box_filter(input: torch.Tensor, channel: int):
    if input.shape[1] % channel != 0:
        return None

    kernel_size_sq = input.shape[1] // channel
    # input.shape[1] == kernel_size_sq * channel

    filters = torch.zeros(
        channel, input.shape[1], device=input.device, dtype=input.dtype
    )

    for i in range(channel):
        filters[
            i, (kernel_size_sq * i) : (kernel_size_sq * (i + 1))  # noqa: E203
        ] = (1.0 / kernel_size_sq)

    return filters @ input
