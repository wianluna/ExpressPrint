import os
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from expressprint.datasets import ImageNetDataLoader


class TestImageNetDataLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = Path(os.getenv('IMAGENET_DIR', '../../imagenet/ILSVRC/Data/CLS-LOC'))
        cls.data_loader = ImageNetDataLoader(cls.data_dir)
        
        cls.train_transforms = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        cls.val_transforms = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

    def test_data_loader_initialization(self):
        self.assertIsInstance(self.data_loader, ImageNetDataLoader)
        self.assertEqual(self.data_loader.data_dir, self.data_dir)

    def test_train_loader(self):
        train_loader = self.data_loader.get_train_loader(
            batch_size=32, 
            train_transforms=self.train_transforms
        )
        self.assertTrue(len(train_loader) > 0)
        
        # Test batch shapes
        images, labels = next(iter(train_loader))
        self.assertEqual(images.shape[0], 32)  # batch size
        self.assertEqual(images.shape[1], 3)   # channels
        self.assertEqual(images.shape[2], 512) # height
        self.assertEqual(images.shape[3], 512) # width
        self.assertEqual(labels.shape[0], 32)  # batch size

    def test_val_loader(self):
        val_loader = self.data_loader.get_val_loader(
            batch_size=32, 
            val_transforms=self.val_transforms
        )
        self.assertTrue(len(val_loader) > 0)
        
        # Test batch shapes
        images, labels = next(iter(val_loader))
        self.assertEqual(images.shape[0], 32)  # batch size
        self.assertEqual(images.shape[1], 3)   # channels
        self.assertEqual(images.shape[2], 512) # height
        self.assertEqual(images.shape[3], 512) # width
        self.assertEqual(labels.shape[0], 32)  # batch size

    def test_sample_visualization(self):
        train_loader = self.data_loader.get_train_loader(
            batch_size=32, 
            train_transforms=self.train_transforms
        )
        
        images, labels = next(iter(train_loader))
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        images = images.cpu()

        for i in range(2):
            for j in range(5):
                idx = i * 5 + j
                if idx < len(images):
                    img = images[idx].permute(1, 2, 0)
                    axes[i, j].imshow(img)
                    axes[i, j].axis('off')
                    axes[i, j].set_title(f'Class: {labels[idx].item()}')

        plt.tight_layout()
        save_path = Path("assets/imagenet_sample_batch.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        
        self.assertTrue(save_path.exists())


if __name__ == '__main__':
    unittest.main()
