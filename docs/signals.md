# Signals

**shared parameters**:
- amplitude: $A$
- frequency: $f_0$
## sineburst
$$f(t)=A\,\sin\left(2\pi f_0t\right)\sin^2\left(\frac{\pi f_0t}{n}\right)\qquad\textrm{for }0<t\le\frac{n}{f_0},\qquad0\textrm{ otherwise}$$
with
- cycles: $n$
## ricker
$$f(t)=A\left(1-2\left(\pi f_0\left(t-t_0\right)\right)^2\right)\exp\left(-\left(\pi f_0\left(t-t_0\right)\right)^2\right),\qquad t_0=\frac{1}{f_0}\textrm{ by default}$$
with
- delay: $t_0$
## gabor
$$f(t)=A\exp\left(-\left(\frac{t-t_0}{\tau}\right)^2\right)\sin\left(2\pi f_0\left(t-t_0\right)\right),\qquad\tau=\frac{n}{2f_0},\qquad t_0=3\tau\textrm{ by default}$$
with
- cycles: $n$, the periods spanning the $1/e$ width of the envelope
- delay: $t_0$

The bandwidth that `sineburst` and `ricker` fix by construction, as a knob: large $n$ is a
narrowband burst and $n\approx1$ is as broad as a `ricker`, which is what lets one run
cover every wavelength an `intensity` objective scores. The sine phase is odd about $t_0$,
so the mean vanishes and a curl-curl [maxwell](maxwell.md) run keeps no static blob, and
the default delay holds the truncation at $t=0$ to $10^{-4}$ of the peak
## chirp
$$f(t)=A\,w(t)\sin\left(2\pi t\left(f_1+\frac{f_2-f_1}{2T}t\right)\right)\qquad\textrm{for }0<t\le T,\qquad0\textrm{ otherwise}$$
$$w(t)=\sin^2\left(\frac{\pi}{2}\min\left(\frac{2\min\left(t,T-t\right)}{\rho T},1\right)\right)$$
with
- band: $(f_1,f_2)$, swept linearly
- duration: $T$
- taper: $\rho$, the share of $T$ spent ramping, split between the two ends

Flat where the Gaussian family is peaked, so every frequency the objective scores is driven
equally hard rather than at whatever the envelope happens to leave there. The window is
Tukey and not `sineburst`'s Hann because a full-duration rise attenuates the start and the
end of the sweep, which are exactly the band edges it is there to cover: $\rho$ trades those
edges against the ringing a sharper cutoff spreads across the band

Nothing here is free of the support it needs. `chirp` buys its flat band by running many
periods longer than a `gabor` of the same width, and the stored-history
[sensitivity](sensitivity.md) pays for every one of those steps, so the flatness is worth
buying only when several frequencies are scored at once
