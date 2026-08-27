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
