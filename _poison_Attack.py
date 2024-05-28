import numpy as np
import torch
import matplotlib.pyplot as plt

def poison_labels(dataset, poison_fraction=0.1, target_label=0):
    """ Poison a fraction of the dataset by changing the labels to a specific target label. """
    num_poisoned = int(len(dataset) * poison_fraction)
    indices = np.random.choice(len(dataset), num_poisoned, replace=False)
    for idx in indices:
        _, original_label = dataset[idx]
        dataset.targets[idx] = target_label
        print(f"Changed index {idx} from {original_label} to {target_label}")
        
def add_noise_to_images(dataset, fraction=0.1, noise_level=0.2):
    """ Add random noise to a fraction of the images in the dataset. """
    num_poisoned = int(len(dataset) * fraction)
    indices = np.random.choice(len(dataset), num_poisoned, replace=False)
    for idx in indices:
        image, label = dataset[idx]
        noise = torch.randn_like(image) * noise_level
        poisoned_image = image + noise
        dataset.data[idx] = (poisoned_image.clamp(0, 1) * 255).byte()  # Clamp to valid image range and convert back
 