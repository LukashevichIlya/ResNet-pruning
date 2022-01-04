import unittest

import torch

from resnet import create_resnet20


class TestResNet20(unittest.TestCase):
    def setUp(self) -> None:
        self.model = create_resnet20()
        torch.manual_seed(0)
        self.x = torch.randn((3, 3, 32, 32))

    def test_forward_numel(self):
        self.assertEqual(self.model(self.x).shape.numel(), 30)

    def test_forward_shape(self):
        self.assertEqual(list(self.model(self.x).shape), [3, 10])


if __name__ == '__main__':
    unittest.main(verbosity=10)
