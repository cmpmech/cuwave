"""What the generator in `cuwave.nn` promises its callers, pinned on the cpu.

Autograd differentiates it, so there is no hand-written adjoint left to check
numerically. What can still go wrong is the wiring: a level built into a `ModuleList`
and never reached by `forward`, a stack that silently assumes two dimensions, an output
that does not start where the pixel-wise drivers start, or a latent that is learnable
when it was not asked to be. So each test pins a contract a driver relies on, and
`GradientTest` walks every parameter to catch the branch that was built but not used.

torch is an optional dependency of the package (`pip install -e ".[nn]"`), hence the
skip; none of this needs a GPU.
"""

import unittest

try:
    import torch
    from torch import nn

    HAS_TORCH = True
except ImportError:  # torch not installed
    HAS_TORCH = False

if HAS_TORCH:
    from cuwave.nn import Generator

# one shape per dimensionality, coarse enough that three levels still divide it
SHAPES = {1: (32,), 2: (16, 16), 3: (8, 8, 8)}


@unittest.skipUnless(HAS_TORCH, "requires torch")
class GeneratorTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_every_dimension(self):
        for dim, shape in SHAPES.items():
            generator = Generator([4, 2, 1], shape, kernel=3, dim=dim)
            coarse = tuple(n // 4 for n in shape)  # two upsamplings
            self.assertEqual(tuple(generator.latent.shape), (1, 4, *coarse))
            self.assertEqual(tuple(generator().shape), (1, 1, *shape))

    def test_the_field_starts_undamaged(self):
        generator = Generator([8, 4, 2, 1], (64, 64))
        with torch.no_grad():
            field = generator()
        # a large positive bias into the sigmoid starts the field flat and undamaged
        self.assertGreater(float(field.min()), 0.999)
        self.assertLessEqual(float(field.max()), 1.0)

    def test_the_output_bias_is_the_knob(self):
        with torch.no_grad():
            field = Generator([4, 2, 1], (16, 16), output_bias=0.0)()
        self.assertLess(float((field - 0.5).abs().max()), 0.1)

    def test_the_latent_is_a_buffer_unless_it_is_learnable(self):
        fixed = Generator([4, 2, 1], (16, 16))
        self.assertIn("latent", dict(fixed.named_buffers()))
        self.assertNotIn("latent", dict(fixed.named_parameters()))

        learnable = Generator([4, 2, 1], (16, 16), learnable=True)
        self.assertIn("latent", dict(learnable.named_parameters()))
        learnable().sum().backward()
        self.assertTrue(bool(learnable.latent.grad.any()))

    def test_the_activation_is_a_knob(self):
        for activation in (nn.ReLU, nn.LeakyReLU, nn.GELU):
            generator = Generator([4, 2, 1], (16, 16), kernel=3, activation=activation)
            self.assertEqual(tuple(generator().shape), (1, 1, 16, 16))

    def test_the_shape_is_rejected_when_it_does_not_fit(self):
        with self.assertRaises(ValueError):  # not divisible by the upsamplings
            Generator([4, 2, 1], (10, 16))
        with self.assertRaises(ValueError):  # two sizes for a 3D stack
            Generator([4, 2, 1], (16, 16), dim=3)
        with self.assertRaises(ValueError):  # no 4D convolution
            Generator([4, 2, 1], (16, 16, 16, 16), dim=4)


@unittest.skipUnless(HAS_TORCH, "requires torch")
class GradientTest(unittest.TestCase):
    def check(self, model, *args):
        torch.manual_seed(0)
        model(*args).sum().backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, f"{name} is not in the graph")
            self.assertTrue(bool(parameter.grad.any()), f"{name} gets a zero gradient")

    def test_the_generator_reaches_every_weight(self):
        self.check(Generator([8, 4, 2, 1], (32, 32), learnable=True))


if __name__ == "__main__":
    unittest.main()
