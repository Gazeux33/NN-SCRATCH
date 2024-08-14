import unittest
import numpy as np
from nn.Dataloader import Dataloader


class TestDataloader(unittest.TestCase):

    def setUp(self):
        self.dataset = np.random.randn(100, 10)
        self.target = np.random.randint(0, 2, 100)
        self.batch_size = 5
        self.x_dataloader = Dataloader(self.dataset, batch_size=self.batch_size)
        self.xy_dataloader = Dataloader(self.dataset, self.target, batch_size=self.batch_size)

    def test_dataloader_len(self):
        self.assertEqual(len(self.x_dataloader), 20)

    def test_x_dataloader_iter(self):
        i = 0
        for i, x in enumerate(self.x_dataloader):
            self.assertEqual(x.shape, (5, 10))
        self.assertEqual(i, 19)

    def test_xy_dataloader_iter(self):
        i = 0
        for i, (x, y) in enumerate(self.xy_dataloader):
            self.assertEqual(x.shape, (5, 10))
            self.assertEqual(y.shape, (5,1))
        self.assertEqual(i, 19)

    def test_overbatch(self):
        data = np.random.randn(103, 10)
        batch_size = 5
        dataloader = Dataloader(data,batch_size=batch_size)

        self.assertEqual(len(dataloader), 20)
        i = 0
        for i, x in enumerate(self.x_dataloader):
            pass
        self.assertEqual(i, 19)


if __name__ == '__main__':
    unittest.main()
