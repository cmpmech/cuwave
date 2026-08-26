import numpy as np


def sineburst(t, amplitude, frequency, cycles):
    """Hann-windowed sine burst of `cycles` periods at `frequency`, zero outside its support."""
    mask = (t > 0) & (t <= cycles / frequency)
    return (
        amplitude
        * mask
        * np.sin(2 * np.pi * frequency * t)
        * np.sin(np.pi * frequency * t / cycles) ** 2
    )


def ricker(t, amplitude, frequency, delay=None):
    """Ricker wavelet at `frequency`, centered at `delay` (default one period)."""
    if delay is None:
        delay = 1.0 / frequency
    arg = (np.pi * frequency * (t - delay)) ** 2
    return amplitude * (1.0 - 2.0 * arg) * np.exp(-arg)
