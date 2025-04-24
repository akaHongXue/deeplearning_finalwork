import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import tarfile
import urllib.request
from multiprocessing import Pool
import cupy as cp
from skimage.transform import rotate, AffineTransform, warp
import random
from dataclasses import dataclass
from typing import Tuple, Optional

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_DIR = "../cifar-10-batches-py"


@dataclass
class AugmentationConfig:
    flip: bool = True
    rotation: float = 15.0
    translate: float = 2.0
    brightness: Tuple[float, float] = (0.8, 1.2)
    crop: bool = True
    color_jitter: bool = True
    jitter_strength: float = 0.1


def download_and_extract_cifar10():
    if not os.path.exists(CIFAR10_DIR):
        print("Downloading CIFAR-10 dataset...")
        tar_path = "cifar-10-python.tar.gz"
        urllib.request.urlretrieve(CIFAR10_URL, tar_path)
        with tarfile.open(tar_path) as tar:
            tar.extractall()
        os.remove(tar_path)
        print("Download and extraction complete!")


def load_cifar10_batch(batch_file):
    with open(batch_file, 'rb') as f:
        batch_data = pickle.load(f, encoding='bytes')
    data = batch_data[b'data']
    labels = batch_data[b'labels']
    data = data.reshape((len(data), 3, 32, 32)).astype(np.float32)
    data = np.transpose(data, (0, 2, 3, 1))  # Convert to NHWC format
    return data, np.array(labels)


def color_jitter(image, strength=0.1):
    # Random color jitter
    jitter = np.random.uniform(-strength, strength, 3)
    image = image + jitter
    return np.clip(image, 0, 1)


def random_crop(image, crop_size=32):
    h, w = image.shape[:2]
    if h <= crop_size or w <= crop_size:
        return image
    top = np.random.randint(0, h - crop_size)
    left = np.random.randint(0, w - crop_size)
    return image[top:top + crop_size, left:left + crop_size]


def augment_image(image, config=AugmentationConfig()):
    """Apply random augmentations to a single image"""
    # Random horizontal flip
    if config.flip and random.random() > 0.5:
        image = image[:, ::-1, :]

    # Random rotation
    if config.rotation > 0:
        angle = random.uniform(-config.rotation, config.rotation)
        image = rotate(image, angle, mode='reflect')

    # Random translation
    if config.translate > 0:
        tx, ty = random.uniform(-config.translate, config.translate), \
            random.uniform(-config.translate, config.translate)
        tf = AffineTransform(translation=(tx, ty))
        image = warp(image, tf, mode='reflect')

    # Random crop
    if config.crop:
        image = random_crop(image)

    # Color jitter
    if config.color_jitter:
        image = color_jitter(image, config.jitter_strength)

    # Random brightness adjustment
    if config.brightness:
        image = np.clip(image * random.uniform(*config.brightness), 0, 1)

    return image


def load_batches_parallel(batch_files):
    with Pool(min(os.cpu_count(), len(batch_files))) as p:
        results = p.map(load_cifar10_batch, batch_files)
    return results


def filter_cat_dog(data, labels):
    mask = np.isin(labels, [3, 5])  # 3=cat, 5=dog
    filtered_data = data[mask]
    filtered_labels = labels[mask]
    filtered_labels = np.where(filtered_labels == 3, 0, 1)  # cat=0, dog=1
    return filtered_data, filtered_labels


def to_onehot(y, num_classes):
    return np.eye(num_classes)[y]


class CIFAR10Loader:
    def __init__(self, val_size=0.2, random_state=42, prefetch=True,
                 filter_cats_dogs=False, augment=True,
                 augmentation_config=AugmentationConfig(),
                 release_cpu_mem=True):
        self.prefetch = prefetch
        self.filter_cats_dogs = filter_cats_dogs
        self.augment = augment
        self.augmentation_config = augmentation_config
        self.release_cpu_mem = release_cpu_mem
        download_and_extract_cifar10()

        # Load data
        train_files = [os.path.join(CIFAR10_DIR, f"data_batch_{i}") for i in range(1, 6)]
        train_results = load_batches_parallel(train_files)
        self.x_train = np.concatenate([r[0] for r in train_results], axis=0)
        self.y_train = np.concatenate([r[1] for r in train_results], axis=0)

        test_file = os.path.join(CIFAR10_DIR, "test_batch")
        self.x_test, self.y_test = load_cifar10_batch(test_file)

        # Filter cat and dog classes
        if self.filter_cats_dogs:
            self.x_train, self.y_train = filter_cat_dog(self.x_train, self.y_train)
            self.x_test, self.y_test = filter_cat_dog(self.x_test, self.y_test)
            self.num_classes = 2
        else:
            self.num_classes = 10

        # Split train/validation
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            self.x_train, self.y_train,
            test_size=val_size,
            random_state=random_state,
            stratify=self.y_train
        )

        # Apply data augmentation to training set
        if self.augment:
            augmented_images = []
            augmented_labels = []
            for img, label in zip(self.x_train, self.y_train):
                augmented_images.append(augment_image(img, self.augmentation_config))
                augmented_labels.append(label)
                # Add original image as well
                augmented_images.append(img)
                augmented_labels.append(label)

            self.x_train = np.array(augmented_images)
            self.y_train = np.array(augmented_labels)

        # Normalize
        self.x_train = self.x_train / 255.0
        self.x_val = self.x_val / 255.0
        self.x_test = self.x_test / 255.0

        # Convert to one-hot
        self.y_train = to_onehot(self.y_train, self.num_classes)
        self.y_val = to_onehot(self.y_val, self.num_classes)
        self.y_test = to_onehot(self.y_test, self.num_classes)

        # Prefetch to GPU
        if self.prefetch:
            print("\nPrefetching data to GPU...")
            self.x_train_gpu = cp.asarray(self.x_train, dtype=cp.float32)
            self.y_train_gpu = cp.asarray(self.y_train, dtype=cp.float32)
            self.x_val_gpu = cp.asarray(self.x_val, dtype=cp.float32)
            self.y_val_gpu = cp.asarray(self.y_val, dtype=cp.float32)
            self.x_test_gpu = cp.asarray(self.x_test, dtype=cp.float32)
            self.y_test_gpu = cp.asarray(self.y_test, dtype=cp.float32)

            if self.release_cpu_mem:
                del self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test
                self.x_train = self.y_train = self.x_val = self.y_val = self.x_test = self.y_test = None

    def get_data(self, device='gpu'):
        if device == 'gpu' and self.prefetch:
            return (self.x_train_gpu, self.y_train_gpu), \
                (self.x_val_gpu, self.y_val_gpu), \
                (self.x_test_gpu, self.y_test_gpu)
        else:
            if self.x_train is None:
                raise RuntimeError(
                    "CPU data has been released. Initialize with release_cpu_mem=False to keep CPU data.")
            return (self.x_train, self.y_train), \
                (self.x_val, self.y_val), \
                (self.x_test, self.y_test)