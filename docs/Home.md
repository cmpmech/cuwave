# CuWave

**Forward simulation** of the following wave equations:

$$\textrm{pressure:}\qquad m\ddot{u}+d\dot{u}-\nabla\cdot(k\nabla u)=f$$
$$\textrm{elastic:}\qquad m\ddot{\mathbf{u}}+d\dot{\mathbf{u}}-\nabla\cdot\boldsymbol{\sigma}=\mathbf{f},\qquad\boldsymbol{\sigma}=\lambda\,\textrm{tr}\left(\boldsymbol{\varepsilon}\right)\mathbf{I}+2\mu\boldsymbol{\varepsilon},\qquad\boldsymbol{\varepsilon}=\frac{1}{2}\left(\nabla\mathbf{u}+\nabla\mathbf{u}^\top\right)$$
$$\textrm{electromagnetic:}\qquad\varepsilon\ddot{\mathbf{E}}+\sigma\dot{\mathbf{E}}+\nabla\times\left(\frac{1}{\mu}\nabla\times\mathbf{E}\right)=-\dot{\mathbf{J}}$$


**Differentiation** through the simulation w.r.t. the inertia $m$ and/or stiffness $k$ coefficients of any of the above ($\varepsilon$ and $1/\mu$ in the electromagnetic case) enabled through cost function $C(u)$ (and derivative $dC(u)/du$) depending on forward solution $u$

Gradients useful in **optimization** scenarios such as inverse problems (full waveform inversion) or structural optimization (transient acoustic and photonic topology optimization)

Common **helpers** ease the creation of driver files for applied problem setups

See [install](install.md) for the installation

## forward simulation

- [wave](wave.md), the grid and the time loop every equation shares
- [scalar](scalar.md) with [forward CUDA](cuda_scalar.md) kernels
- [elastic](elastic.md) with [forward CUDA](cuda_elastic.md) kernels
- [anisotropic](anisotropic.md) with [forward CUDA](cuda_anisotropic.md) kernels
- [maxwell](maxwell.md) with [forward CUDA](cuda_maxwell.md) kernels, the 2D reductions running on the scalar ones
- [stencils](stencils.md)
- [boundary](boundary.md)

## sensitivity analysis

- [sensitivity](sensitivity.md) with [scalar backward CUDA](cuda_scalar_sensitivity.md), [elastic backward CUDA](cuda_elastic_sensitivity.md), [anisotropic backward CUDA](cuda_anisotropic_sensitivity.md) and [maxwell backward CUDA](cuda_maxwell_sensitivity.md) kernels

## optimization

- [regularization](regularization.md)
- [neural networks](nn.md)
- [optimization](optimization.md)

## applications

- [full waveform inversion](fwi.md)
- [source inversion](source_inversion.md)
- [transient acoustic topology optimization](tato.md)
- [transient photonic topology optimization](tpto.md)

## problem helpers

- [signals](signals.md)
- [geometry](geometry.md)
- [utils](utils.md)
- [evals](evals.md)
- [postprocessing](postprocessing.md)
