# CuWave

**forward simulation** of the following wave equations:
$$\textrm{pressure:}\qquad m\ddot{u}+d\dot{u}-\nabla\cdot(k\nabla u)=f$$
$$\textrm{elastic:}\qquad m\ddot{\mathbf{u}}+d\dot{\mathbf{u}}-\nabla\cdot\boldsymbol{\sigma}=\mathbf{f},\qquad\boldsymbol{\sigma}=\lambda\,\textrm{tr}\left(\boldsymbol{\varepsilon}\right)\mathbf{I}+2\mu\boldsymbol{\varepsilon},\qquad\boldsymbol{\varepsilon}=\frac{1}{2}\left(\nabla\mathbf{u}+\nabla\mathbf{u}^\top\right)$$
**Differentiation** through the simulation w.r.t. $m$ and/or $k$ enabled through cost function $C(u)$ (and derivative $dC(u)/du$) depending on forward solution $u$

Gradients useful in **optimization** scenarios such as inverse problems (full waveform inversion: fwi) or structural optimization (transient acoustic topology optimization: tato)

Common **helpers** ease the creation of driver files for applied problem setups
## forward simulation
- [wave](wave.md) with [CUDA](cuda.md) kernels
- [stencils](stencils.md)
- [integrators](integrators.md) 
- [boundary](boundary.md)
## sensitivity analysis
- [sensitivity](sensitivity.md) with [CUDA](cuda.md) kernels
## optimization
- [regularization](regularization.md)
- [optimization](optimization.md)
- [uncertainty](uncertainty.md)
## problem helpers
- [signals](signals.md)
- [geometry](geometry.md)
- [fwi](fwi.md)
- [tato](tato.md)
- [utils](utils.md) 