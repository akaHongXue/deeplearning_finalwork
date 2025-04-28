from data import CIFAR10Loader
from network import SequentialLayer, ConvLayer, LeakyReLULayer, MaxPoolingLayer, FlattenLayer, DenseLayer, Adam, \
    CrossEntropyLoss, create_config
from train import train, build_model


def create_network():
    """Define the CNN architecture."""
    return SequentialLayer([
        ConvLayer(3, 32, 3, 1, 1),
        LeakyReLULayer(),
        MaxPoolingLayer(),
        ConvLayer(32, 64, 3, 1, 1),
        LeakyReLULayer(),
        MaxPoolingLayer(),
        ConvLayer(64, 64, 3, 1, 1),
        LeakyReLULayer(),
        MaxPoolingLayer(),
        FlattenLayer(),
        DenseLayer(4 * 4 * 64, 32),
        LeakyReLULayer(),
        DenseLayer(32, 2)
    ])


def load_dataset():
    """Load and preprocess CIFAR-10 (cats vs. dogs)."""
    loader = CIFAR10Loader(prefetch=True, filter_cats_dogs=True, augment=True)
    return loader.get_data('gpu')

def main():
    print(" Initializing...")
    network = create_network()
    data = load_dataset()
    config = create_config(network, data)
    history, test_acc = train(config)
    print(f"\n Done! Test Accuracy: {test_acc:.2%}")


if __name__ == '__main__':
    main()
