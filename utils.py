import numpy as np
import torch
import matplotlib.pyplot as plt

# modify ImageFolder's label
def poison_labels(dataset, indices, poison_fraction=0.1, target_label=0):
    """ Poison a fraction of the dataset by changing the labels to a specific target label. """
    num_poisoned = int(len(indices) * poison_fraction)
    poisoned_indices = np.random.choice(indices, num_poisoned, replace=False)
    for idx in poisoned_indices:
        original_label = dataset.dataset.targets[idx]  # Access the original dataset's targets
        dataset.dataset.targets[idx] = target_label
        print(f"Changed index {idx} from {original_label} to {target_label}")


# add noise to images, by custom dataset
class NoisyDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, fraction=0.1, noise_level=0.2):
        self.base_dataset = base_dataset
        self.fraction = fraction
        self.noise_level = noise_level
        self.noise_indices = set(np.random.choice(len(self.base_dataset), int(len(self.base_dataset) * fraction), replace=False))

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        if idx in self.noise_indices:
            noise = torch.randn_like(image) * self.noise_level
            image = image + noise
            image = image.clamp(0, 1)  # Clamp to ensure the image is still valid
        return image, label