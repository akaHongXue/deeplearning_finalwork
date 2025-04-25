from data import CIFAR10Loader
from network import SequentialLayer, ConvLayer, LeakyReLULayer, MaxPoolingLayer,FlattenLayer, DenseLayer, Adam, CrossEntropyLoss
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


def create_config(network, data, epochs=200, batch_size=64, lr=0.001, patience=40):
    """Build the training configuration."""
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data
    return build_model(
        network=network,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        optimizer=Adam,
        loss_function=CrossEntropyLoss,
        patience=patience
    )


def main():
    print(" Initializing...")
    network = create_network()
    data = load_dataset()
    config = create_config(network, data)
    history, test_acc = train(config)
    print(f"\n Done! Test Accuracy: {test_acc:.2%}")


if __name__ == '__main__':
    main()
