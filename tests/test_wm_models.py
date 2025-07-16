import unittest

import torch

from expressprint.models import WMDecoder, WMEncoder


class TestWatermarkModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        cls.batch_size = 100
        cls.watermark_size = 32
        cls.feature_dim = 1024

        cls.encoder = WMEncoder(watermark_size=cls.watermark_size, feature_dim=cls.feature_dim)
        cls.decoder = WMDecoder(watermark_size=cls.watermark_size, feature_dim=cls.feature_dim)

    def test_encoder_decoder_shapes(self):
        """Test that encoder and decoder produce expected output shapes."""
        x = torch.randn(self.batch_size, self.feature_dim)
        binary_message = torch.randint(0, 2, (self.batch_size, self.watermark_size))
        encoded = self.encoder(x, binary_message)
        decoded_message = self.decoder(encoded)

        self.assertEqual(binary_message.shape, (self.batch_size, self.watermark_size))
        self.assertEqual(encoded.shape[0], self.batch_size)  # Batch dimension preserved
        self.assertEqual(decoded_message.shape, (self.batch_size, self.watermark_size))


if __name__ == "__main__":
    unittest.main()
