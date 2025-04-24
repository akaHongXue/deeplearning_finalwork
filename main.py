from data import CIFAR10Loader
from network import SequentialLayer, ConvLayer, LeakyReLULayer, MaxPoolingLayer, FlattenLayer, DenseLayer, Adam, CrossEntropyLoss
from new.network import SGD
from train import train, build_model

if __name__ == '__main__':
    # Hyperparameters
    EPOCHS = 200
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0001

    # Improved network architecture
    network = SequentialLayer([
        ConvLayer(3,32,3,1,1),  # 初始通道数减半
        LeakyReLULayer(),
        MaxPoolingLayer(),
        ConvLayer(32, 64, 3, 1, 1),
        LeakyReLULayer(),
        MaxPoolingLayer(),
        ConvLayer(64, 64, 3, 1, 1),
        LeakyReLULayer(),
        MaxPoolingLayer(),
        FlattenLayer(),
        DenseLayer(4*4*64,32),
        LeakyReLULayer(),
        DenseLayer(32, 2),

    ])

    # Load data with augmentation
    data_loader = CIFAR10Loader(prefetch=True, filter_cats_dogs=True, augment=True)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data_loader.get_data('gpu')

    # Build model with AdamW optimizer and label smoothing
    model = build_model(
        network=network,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        optimizer=Adam,
        loss_function=CrossEntropyLoss,
        patience=40 # Increased patience for early stopping
    )

    # Start training
    train(model)