import numpy as np
import numpy.typing as npt


def sineburst(
    t: npt.NDArray[np.float64], amplitude: float, frequency: float, cycles: int
) -> npt.NDArray[np.float64]:
    """Hann-windowed sine burst of `cycles` periods at `frequency`, zero outside its support."""
    mask = (t > 0) & (t <= cycles / frequency)
    return (
        amplitude
        * mask
        * np.sin(2 * np.pi * frequency * t)
        * np.sin(np.pi * frequency * t / cycles) ** 2
    )


def ricker(
    t: npt.NDArray[np.float64],
    amplitude: float,
    frequency: float,
    delay: float | None = None,
) -> npt.NDArray[np.float64]:
    """Ricker wavelet at `frequency`, centered at `delay` (default one period)."""
    if delay is None:
        delay = 1.0 / frequency
    arg = (np.pi * frequency * (t - delay)) ** 2
    return amplitude * (1.0 - 2.0 * arg) * np.exp(-arg)


def gabor(
    t: npt.NDArray[np.float64],
    amplitude: float,
    frequency: float,
    cycles: float,
    delay: float | None = None,
) -> npt.NDArray[np.float64]:
    """Gaussian-modulated sine at `frequency`, its 1/e envelope `cycles` periods wide.

    Odd about `delay`, three envelope widths in by default, so the mean is zero.
    """
    tau = 0.5 * cycles / frequency
    if delay is None:
        delay = 3.0 * tau
    lag = t - delay
    return amplitude * np.exp(-((lag / tau) ** 2)) * np.sin(2 * np.pi * frequency * lag)


def chirp(
    t: npt.NDArray[np.float64],
    amplitude: float,
    band: tuple[float, float],
    duration: float,
    taper: float = 0.1,
) -> npt.NDArray[np.float64]:
    """Linear sweep across `band` over `duration`, zero outside it.

    `taper` is the share of `duration` spent ramping, split between the two ends. A
    Tukey window and not `sineburst`'s Hann, whose rise would attenuate the band edges
    the sweep is there to cover flatly.
    """
    mask = (t > 0) & (t <= duration)
    edge = 0.5 * taper * duration
    window = np.sin(0.5 * np.pi * np.clip(np.minimum(t, duration - t) / edge, 0.0, 1.0))
    rate = (band[1] - band[0]) / duration
    return (
        amplitude
        * mask
        * np.sin(2 * np.pi * t * (band[0] + 0.5 * rate * t))
        * window**2
    )
