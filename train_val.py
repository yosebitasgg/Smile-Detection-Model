import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt


class TrainVal:
    def __init__(self, lfw_dataset: Dataset, val_split: float = 0.2):
        """
        lfw_dataset: instancia de LFWDataset ya creada
        val_split: proporción para validación (0.2 = 20%)
        """
        self.full_dataset = lfw_dataset
        self.val_split = val_split

        # crear train y val
        self.train_dataset, self.val_dataset = self._split_dataset()

    def _split_dataset(self):
        val_size = int(len(self.full_dataset) * self.val_split)
        train_size = len(self.full_dataset) - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            self.full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        return train_dataset, val_dataset

    def visualize_samples(self, n_por_clase: int = 5):
        """Muestra n imágenes smiling y n non-smiling."""
        smiling_count = 0
        non_smiling_count = 0

        plt.figure(figsize=(10, 4))
        for i in range(len(self.full_dataset)):
            image, label = self.full_dataset[i]

            if label == 1 and smiling_count < n_por_clase:
                plt.subplot(2, n_por_clase, smiling_count + 1)
                plt.imshow(image.permute(1, 2, 0))
                plt.title("Smiling")
                plt.axis('off')
                smiling_count += 1

            elif label == 0 and non_smiling_count < n_por_clase:
                plt.subplot(2, n_por_clase, non_smiling_count + 1 + n_por_clase)
                plt.imshow(image.permute(1, 2, 0))
                plt.title("Non-Smiling")
                plt.axis('off')
                non_smiling_count += 1

            if smiling_count >= n_por_clase and non_smiling_count >= n_por_clase:
                break

        plt.tight_layout()
        plt.show()
