# Neural networks

**Neural networks** reparametrize the design field as the output of a convolutional generator, so the optimizer updates network weights instead of nodal values, following [Herrmann, Bürchner, Dietrich & Kollmannsberger 2023](https://doi.org/10.1016/j.cma.2023.116307)

The design is then $\gamma=G_\theta(z)$ with the fixed latent `latent` $z$ and the weights `parameters` $\theta$, and the chain rule the driver already applies for a filter or a projection extends by one more factor — the gradient with respect to $\gamma$ is handed to `backward` and PyTorch produces $\partial\gamma/\partial\theta$. This is the only module that uses torch; the solver and its adjoints are cupy throughout

| member | signature | description |
|---|---|---|
| constructor | `Generator(channels, shape, kernel=5, activation=nn.GELU, dim=2, output_bias=10.0, learnable=False)` | stacks one upsampling block per channel transition, doubling the resolution each time, and ends in a sigmoid so the field lands in $[0,1]$ |
| forward pass | `forward()` | maps the stored latent to the field, taking no argument since the input is fixed |
| size | `nn_params(model)` | trainable parameter count, which is the dimension the optimization actually runs in |

`shape` must be divisible by $2^{L-1}$ for `channels` of length $L$, since each transition doubles the resolution, and a mismatch raises rather than silently cropping

The output convolution starts from small random weights and a large `output_bias`, so the generator begins flat at 1 — the undamaged field the pixel-wise drivers also start from — rather than at whatever a Xavier initialization happens to produce. `latent` is a registered buffer, so it moves with the model but is not optimized; `learnable=True` promotes it to a parameter and lets the optimizer reshape the input as well as the map

The reparametrization is a regularization by architecture rather than by penalty: the generator can only express fields its convolutions can build, which suppresses the node-scale noise a pixel-wise inversion accumulates, and the price is that the reachable set is no longer all fields and the optimization is no longer convex in the variables it steps
