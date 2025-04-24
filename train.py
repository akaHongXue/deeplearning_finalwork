import cupy as cp
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Callable, Tuple, Dict, Any
import threading
import queue
import numpy as np

from new.network import AMPWrapper


@dataclass
class TrainModelConfig:
    network: 'SequentialLayer'
    x_train: cp.ndarray
    y_train: cp.ndarray
    x_test: cp.ndarray
    y_test: cp.ndarray
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: Callable
    x_val: Optional[cp.ndarray] = None
    y_val: Optional[cp.ndarray] = None
    loss_function: Callable = None
    patience: int = 5
    verbose: bool = True
    use_amp: bool = False
    grad_clip: float = 1.0


class DataPrefetcher:
    def __init__(self, x_data, y_data, batch_size, shuffle=True):
        self.x_data = x_data
        self.y_data = y_data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.queue = queue.Queue(maxsize=3)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._prefetch, daemon=True)
        self.thread.start()

    def _prefetch(self):
        try:
            n_samples = len(self.x_data)
            indices = cp.arange(n_samples)

            while not self.stop_event.is_set():
                if self.shuffle:
                    cp.random.shuffle(indices)

                for start_idx in range(0, n_samples, self.batch_size):
                    batch_indices = indices[start_idx:start_idx + self.batch_size]
                    x_batch = self.x_data[batch_indices]
                    y_batch = self.y_data[batch_indices]
                    self.queue.put((x_batch, y_batch))

                # Epoch结束标志
                self.queue.put((None, None))
        except Exception as e:
            print(f"Prefetch error: {e}")
            self.stop_event.set()

    def __iter__(self):
        return self

    def __next__(self):
        x_batch, y_batch = self.queue.get()
        if x_batch is None:  # Epoch结束
            raise StopIteration
        return x_batch, y_batch

    def __del__(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join()


def build_model(
        network: 'SequentialLayer',
        epochs: int,
        batch_size: int,
        learning_rate: float,
        x_train: cp.ndarray,
        y_train: cp.ndarray,
        x_test: cp.ndarray,
        y_test: cp.ndarray,
        optimizer: Callable,
        loss_function: Callable,
        x_val: Optional[cp.ndarray] = None,
        y_val: Optional[cp.ndarray] = None,
        patience: int = 5,
        use_amp: bool = False,
        grad_clip: float = 1.0
) -> TrainModelConfig:
    return TrainModelConfig(
        network=network,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        optimizer=optimizer,
        loss_function=loss_function,
        x_val=x_val,
        y_val=y_val,
        patience=patience,
        use_amp=use_amp,
        grad_clip=grad_clip
    )


def clip_gradients(grads, max_norm):
    """Clip gradients to prevent explosion"""
    total_norm = cp.sqrt(sum(cp.sum(g ** 2) for g in grads))
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for g in grads:
            g *= clip_coef
    return grads


def train(config: TrainModelConfig):
    model = config.network
    optimizer = config.optimizer(config.learning_rate)
    loss_fn = config.loss_function()

    # Initialize AMP if enabled
    if config.use_amp:
        amp = AMPWrapper(model, optimizer)
    else:
        amp = None

    # 训练记录
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }

    best_val_acc = 0.0
    best_weights = None
    no_improve = 0

    print("\n=== 开始训练 ===")
    print(f"训练样本: {config.x_train.shape[0]} | 验证样本: {config.x_val.shape[0] if config.x_val is not None else 0}")
    print(f"Batch大小: {config.batch_size} | 初始学习率: {config.learning_rate}")

    start_time = time.time()

    for epoch in range(config.epochs):
        epoch_start = time.time()
        model.set_mode(True)

        # 训练阶段
        train_loss = 0.0
        correct = 0
        total = 0

        prefetcher = DataPrefetcher(config.x_train, config.y_train, config.batch_size, shuffle=True)

        for batch_idx, (x_batch, y_batch) in enumerate(prefetcher):
            # 前向传播
            output = model.forward(x_batch)
            loss = loss_fn.forward(output, y_batch)

            # 反向传播
            grad = loss_fn.backward()
            model.backward(grad)

            # ===== 添加的梯度检查代码 =====
            for i, layer in enumerate(model.layers):
                if hasattr(layer, 'dW'):
                    if cp.isnan(layer.dW).any() or cp.isinf(layer.dW).any():
                        print(f"梯度异常在 {i}层 {layer.__class__.__name__} (weights)")
                        print(f"梯度范围: {cp.min(layer.dW)} ~ {cp.max(layer.dW)}")
                if hasattr(layer, 'db'):
                    if cp.isnan(layer.db).any() or cp.isinf(layer.db).any():
                        print(f"梯度异常在 {i}层 {layer.__class__.__name__} (bias)")
                        print(f"梯度范围: {cp.min(layer.db)} ~ {cp.max(layer.db)}")
            # ============================

            # 收集参数和梯度
            params = []
            grads = []
            for layer in model.layers:
                if hasattr(layer, 'weights'):
                    params.append(layer.weights)
                    grads.append(layer.dW)
                if hasattr(layer, 'bias'):
                    params.append(layer.bias)
                    grads.append(layer.db)

            # 参数更新
            optimizer.step(params, grads)

            # 统计
            train_loss += loss
            pred = cp.argmax(output, axis=1)
            correct += cp.sum(pred == cp.argmax(y_batch, axis=1))
            total += len(y_batch)

            # 打印batch信息
            if config.verbose and batch_idx % 10 == 0:
                batch_acc = cp.sum(pred == cp.argmax(y_batch, axis=1)) / len(y_batch)
                print(f"Epoch {epoch + 1}/{config.epochs} | Batch {batch_idx} | "
                      f"Loss: {loss:.4f} | Acc: {batch_acc * 100:.2f}%")

        # 计算epoch指标
        avg_train_loss = train_loss / (batch_idx + 1)
        train_accuracy = correct / total
        history['train_loss'].append(float(avg_train_loss))
        history['train_acc'].append(float(train_accuracy))

        # 验证阶段
        val_accuracy = 0.0
        val_loss = 0.0

        if config.x_val is not None:
            model.set_mode(False)
            val_output = model.forward(config.x_val)
            val_loss = loss_fn.forward(val_output, config.y_val)
            val_pred = cp.argmax(val_output, axis=1)
            val_accuracy = cp.mean(val_pred == cp.argmax(config.y_val, axis=1))

            history['val_loss'].append(float(val_loss))
            history['val_acc'].append(float(val_accuracy))

            # 早停检查
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                no_improve = 0
                # 保存最佳权重
                best_weights = [layer.get_state() if hasattr(layer, 'get_state') else None
                                for layer in model.layers]
            else:
                no_improve += 1
                if no_improve >= config.patience:
                    print(f"\n早停触发！最佳验证准确率: {best_val_acc * 100:.2f}%")
                    break

        # 记录学习率
        if hasattr(optimizer, 'get_lr'):
            history['lr'].append(optimizer.get_lr())
        else:
            history['lr'].append(config.learning_rate)

        # 打印epoch信息
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch + 1}/{config.epochs} | Time: {epoch_time:.1f}s")
        print(f"Train Loss: {avg_train_loss:.4f} | Acc: {train_accuracy * 100:.2f}%")
        if config.x_val is not None:
            print(f"Val Loss: {val_loss:.4f} | Acc: {val_accuracy * 100:.2f}%")
        print("-" * 50)

    # 训练结束
    total_time = time.time() - start_time

    # 恢复最佳权重
    if best_weights is not None:
        for layer, weights in zip(model.layers, best_weights):
            if weights is not None and hasattr(layer, 'set_state'):
                layer.set_state(weights)

    # 测试评估（分批处理）
    model.set_mode(False)
    test_batch_size = 128  # 可以根据GPU内存调整这个值
    test_correct = 0
    test_total = 0

    # 确保测试数据是CuPy数组
    x_test = config.x_test if isinstance(config.x_test, cp.ndarray) else cp.asarray(config.x_test)
    y_test = config.y_test if isinstance(config.y_test, cp.ndarray) else cp.asarray(config.y_test)

    for i in range(0, len(x_test), test_batch_size):
        batch_x = x_test[i:i + test_batch_size]
        batch_y = y_test[i:i + test_batch_size]

        test_output = model.forward(batch_x)
        test_pred = cp.argmax(test_output, axis=1)
        test_correct += cp.sum(test_pred == cp.argmax(batch_y, axis=1))
        test_total += len(batch_y)

        # 可选：释放中间变量内存
        del batch_x, batch_y, test_output, test_pred
        cp.get_default_memory_pool().free_all_blocks()

    test_accuracy = test_correct / test_total

    # 打印结果
    print("\n=== 训练结果 ===")
    print(f"总训练时间: {total_time:.1f}s")
    print(f"最佳验证准确率: {best_val_acc * 100:.2f}%")
    print(f"测试准确率: {test_accuracy * 100:.2f}%")

    # 绘制训练曲线
    if config.verbose:
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(history['train_loss'], label='Train')
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='Validation')
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(history['train_acc'], label='Train')
        if 'val_acc' in history:
            plt.plot(history['val_acc'], label='Validation')
        plt.title('Accuracy Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(history['lr'])
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('LR')

        plt.tight_layout()
        plt.show()

    return history, test_accuracy