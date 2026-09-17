"""Contracts of the source wavelets, which a driver only notices through a bad design.

Both wavelets here trade support against bandwidth, and both halves fail quietly: a
source whose band misses the frequencies `intensity` scores optimizes noise for sixty
iterations, and a source with a nonzero mean leaves a static blob in a curl-curl null
space that no boundary condition removes. So `gabor` is pinned on its mean and on the
bandwidth its `cycles` knob promises, and `chirp` on the flat band it claims to sweep.
"""

import unittest

import numpy as np

from cuwave.signals import chirp, gabor, ricker

DT = 1.0e-3
N = 4096


# -------------------------------------- helpers --------------------------------------
def _spectrum(signal):
    return np.fft.rfftfreq(len(signal), DT), np.abs(np.fft.rfft(signal))


def _band(signal, level=0.5):
    """Frequencies where the magnitude is at least `level` of its peak."""
    f, mag = _spectrum(signal)
    inside = mag >= level * mag.max()
    return f[inside][0], f[inside][-1]


def _width(signal):
    low, high = _band(signal)
    return high - low


class GaborTest(unittest.TestCase):
    def test_the_mean_vanishes_so_a_curl_curl_run_leaves_no_static_blob(self):
        t = np.arange(N) * DT
        for cycles in (1.0, 3.0, 8.0):
            signal = gabor(t, 1.0, 40.0, cycles)
            with self.subTest(cycles=cycles):
                self.assertLess(abs(signal.mean()) / abs(signal).max(), 1e-4)

    def test_the_spectrum_peaks_at_the_carrier(self):
        f, mag = _spectrum(gabor(np.arange(N) * DT, 1.0, 40.0, 5.0))
        self.assertAlmostEqual(f[mag.argmax()], 40.0, delta=f[1] - f[0])

    def test_more_cycles_is_a_narrower_band(self):
        t = np.arange(N) * DT
        self.assertLess(
            _width(gabor(t, 1.0, 40.0, 8.0)), 0.5 * _width(gabor(t, 1.0, 40.0, 2.0))
        )

    def test_one_cycle_is_about_as_broad_as_a_ricker(self):
        t = np.arange(N) * DT
        ratio = _width(gabor(t, 1.0, 40.0, 1.0)) / _width(ricker(t, 1.0, 40.0))
        self.assertAlmostEqual(ratio, 1.0, delta=0.3)

    def test_the_default_delay_keeps_the_truncation_negligible(self):
        signal = gabor(np.arange(N) * DT, 1.0, 40.0, 4.0)
        self.assertLess(abs(signal[0]) / abs(signal).max(), 1e-3)


class ChirpTest(unittest.TestCase):
    def test_the_energy_lands_inside_the_swept_band(self):
        f, mag = _spectrum(chirp(np.arange(N) * DT, 1.0, (20.0, 80.0), 2.0))
        inside = (f >= 20.0) & (f <= 80.0)
        self.assertGreater((mag[inside] ** 2).sum() / (mag**2).sum(), 0.95)

    def test_the_six_db_band_is_the_one_that_was_asked_for(self):
        low, high = _band(chirp(np.arange(N) * DT, 1.0, (20.0, 80.0), 2.0))
        self.assertAlmostEqual(low, 20.0, delta=3.0)
        self.assertAlmostEqual(high, 80.0, delta=3.0)

    def test_the_band_is_flat_where_a_gabor_of_the_same_width_is_peaked(self):
        t = np.arange(N) * DT
        f, swept = _spectrum(chirp(t, 1.0, (20.0, 80.0), 2.0))
        _, peaked = _spectrum(gabor(t, 1.0, 50.0, 2.0))
        inside = (f >= 20.0) & (f <= 80.0)
        self.assertGreater((swept[inside] >= 0.5 * swept.max()).mean(), 0.9)
        self.assertLess((peaked[inside] >= 0.5 * peaked.max()).mean(), 0.6)

    def test_a_wider_taper_trades_band_edges_for_a_cleaner_spectrum(self):
        t = np.arange(N) * DT
        self.assertLess(
            _width(chirp(t, 1.0, (20.0, 80.0), 2.0, 0.3)),
            _width(chirp(t, 1.0, (20.0, 80.0), 2.0, 0.05)),
        )

    def test_it_is_zero_outside_its_support(self):
        t = np.arange(N) * DT
        signal = chirp(t, 1.0, (20.0, 80.0), 2.0)
        self.assertEqual(abs(signal[t > 2.0]).max(), 0.0)


if __name__ == "__main__":
    unittest.main()
