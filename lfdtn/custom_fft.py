import torch
import torch.fft
from asset.utils import normalizeFFT  # We currently use normalizeFFT = false


def custom_fft(t_r):
    """2D FFT an input tensor

    Args:
        t_r -- tensor containing a real 2D signal in its last two dimensions
    """
    t_c = torch.stack([t_r, torch.zeros_like(t_r, requires_grad=False)], dim=-1)
    t_c = torch.view_as_complex(t_c)
    result = torch.fft.fftn(t_c, dim=(-2, -1), norm="forward" if normalizeFFT else "backward")
    return torch.view_as_real(result)


def custom_ifft(t_c):
    """2D FFT an input tensor

    Args:
        t_c -- tensor containing a complex 2D signal in its last three dimensions
    """
    t_c = torch.view_as_complex(t_c)
    result = torch.fft.ifftn(t_c, dim=(-2, -1), norm="forward" if normalizeFFT else "backward")
    return result.real

