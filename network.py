import cupy as cp
import numpy as np
from typing import Optional, Tuple, List, Callable

from cupy import pad

from new.train import build_model


class SequentialLayer:
    def __init__(self, layers):
        self.layers = layers
        self.train_mode = True

    def set_mode(self, is_train):
        self.train_mode = is_train
        for layer in self.layers:
            if hasattr(layer, 'set_mode'):
                layer.set_mode(is_train)

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
            if cp.isnan(inputs).any() or cp.isinf(inputs).any():
                print(f"Numerical instability in {layer.__class__.__name__} layer")
                raise ValueError("Numerical instability")
        return inputs

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                result = layer.backward(grad_output)
                if isinstance(result, tuple):
                    grad_output = result[0]
                else:
                    grad_output = result
            else:
                raise AttributeError(f"{layer.__class__.__name__} missing backward() method")
        return grad_output


class ConvLayer:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.groups = groups
        scale = cp.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weights = cp.random.normal(0, scale, (out_channels, in_channels // groups, kernel_size, kernel_size))
        self.bias = cp.zeros(out_channels)
        self.stride = stride
        self.padding = padding
        self.im2col_kernel = None

    def _im2col(self, x, kernel_size, stride, pad):
        N, C, H, W = x.shape
        kh, kw = kernel_size, kernel_size
        out_h = (H - kh) // stride + 1
        out_w = (W - kw) // stride + 1

        # 不缓存 im2col_kernel，直接按需计算
        im2col_kernel = cp.zeros((N, C, kh, kw, out_h, out_w), dtype=x.dtype)

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride
                w_start = j * stride
                h_end = h_start + kh
                w_end = w_start + kw
                im2col_kernel[:, :, :, :, i, j] = x[:, :, h_start:h_end, w_start:w_end]

        return im2col_kernel.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)

    def forward(self, x):
        if x.ndim == 4 and x.shape[3] <= 3:  # NHWC format
            x = x.transpose(0, 3, 1, 2)  # Convert to NCHW

        self.x = x  # ✅ save input for backward
        self.x_shape = x.shape
        batch, in_c, in_h, in_w = x.shape
        kh, kw = self.weights.shape[2], self.weights.shape[3]
        out_c = self.weights.shape[0]

        out_h = (in_h + 2 * self.padding - kh) // self.stride + 1
        out_w = (in_w + 2 * self.padding - kw) // self.stride + 1

        # ✅ use correct variable for padding
        x_padded = cp.pad(x, [(0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)], mode='constant')

        x_col = self._im2col(x_padded, kh, self.stride, self.padding)
        w_col = self.weights.reshape(self.groups, out_c // self.groups, -1).transpose(1, 0, 2).reshape(out_c, -1)

        out = cp.dot(x_col, w_col.T) + self.bias
        out = out.reshape(batch, out_h, out_w, out_c).transpose(0, 3, 1, 2)

        return out

    def backward(self, dout):
        batch, out_c, out_h, out_w = dout.shape
        in_c = self.x_shape[1]
        kh, kw = self.weights.shape[2], self.weights.shape[3]

        dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, out_c)

        # ✅ correct: pad the actual stored input, not its shape
        x_padded = cp.pad(self.x, [(0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)], mode='constant')
        x_col = self._im2col(x_padded, kh, self.stride, self.padding)

        dw = cp.dot(dout_reshaped.T, x_col).reshape(out_c, in_c // self.groups, kh, kw)
        db = cp.sum(dout_reshaped, axis=0)

        w_reshaped = self.weights.reshape(self.groups, out_c // self.groups, -1).transpose(1, 0, 2).reshape(out_c, -1)
        dx_col = cp.dot(dout_reshaped, w_reshaped)

        dx = cp.zeros((batch, in_c, self.x_shape[2] + 2 * self.padding, self.x_shape[3] + 2 * self.padding), dtype=cp.float32)
        dx_col_reshaped = dx_col.reshape(batch, out_h, out_w, in_c, kh, kw).transpose(0, 3, 4, 5, 1, 2)

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                w_start = j * self.stride
                dx[:, :, h_start:h_start + kh, w_start:w_start + kw] += dx_col_reshaped[:, :, :, :, i, j]
        if self.padding > 0:
            dx = dx[:, :, self.padding:-self.padding, self.padding:-self.padding]
        self.dW = dw
        self.db = db

        return dx, dw, db



class BatchNorm:
    def __init__(self, num_features, momentum=0.9, epsilon=1e-5, affine=True):
        self.gamma = cp.ones(num_features) if affine else None
        self.beta = cp.zeros(num_features) if affine else None
        self.running_mean = cp.zeros(num_features)
        self.running_var = cp.ones(num_features)
        self.momentum = momentum
        self.epsilon = epsilon
        self.train_mode = True
        self.affine = affine

    def forward(self, inputs):
        if inputs.ndim == 4:  # Conv layer
            batch_size, channels, height, width = inputs.shape
            if self.train_mode:
                mean = cp.mean(inputs, axis=(0, 2, 3), keepdims=True)
                var = cp.var(inputs, axis=(0, 2, 3), keepdims=True)
                self.x_hat = (inputs - mean) / cp.sqrt(var + self.epsilon)
                if self.affine:
                    output = self.gamma.reshape(1, channels, 1, 1) * self.x_hat + self.beta.reshape(1, channels, 1, 1)
                else:
                    output = self.x_hat
                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean.squeeze()
                self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var.squeeze()
                self.current_mean = mean
                self.current_var = var
            else:
                output = (inputs - self.running_mean.reshape(1, channels, 1, 1)) / \
                         cp.sqrt(self.running_var.reshape(1, channels, 1, 1) + self.epsilon)
                if self.affine:
                    output = self.gamma.reshape(1, channels, 1, 1) * output + self.beta.reshape(1, channels, 1, 1)
        else:  # FC layer
            if self.train_mode:
                mean = cp.mean(inputs, axis=0)
                var = cp.var(inputs, axis=0)
                self.x_hat = (inputs - mean) / cp.sqrt(var + self.epsilon)
                if self.affine:
                    output = self.gamma * self.x_hat + self.beta
                else:
                    output = self.x_hat
                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
                self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
                self.current_mean = mean
                self.current_var = var
            else:
                output = (inputs - self.running_mean) / cp.sqrt(self.running_var + self.epsilon)
                if self.affine:
                    output = self.gamma * output + self.beta

        self.inputs = inputs
        return output

    def backward(self, gradients):
        if gradients.ndim == 4:  # Conv layer
            batch_size, channels, height, width = gradients.shape
            if self.train_mode:
                if self.affine:
                    d_x_hat = gradients * self.gamma.reshape(1, channels, 1, 1)
                else:
                    d_x_hat = gradients

                d_var = cp.sum(
                    d_x_hat * (self.inputs - self.current_mean) * -0.5 *
                    cp.power(self.current_var + self.epsilon, -1.5),
                    axis=(0, 2, 3), keepdims=True
                )

                d_mean = cp.sum(
                    d_x_hat * -1 / cp.sqrt(self.current_var + self.epsilon),
                    axis=(0, 2, 3), keepdims=True
                )

                d_inputs = d_x_hat / cp.sqrt(self.current_var + self.epsilon) + \
                           d_var * 2 * (self.inputs - self.current_mean) / (batch_size * height * width) + \
                           d_mean / (batch_size * height * width)

                if self.affine:
                    d_gamma = cp.sum(gradients * self.x_hat, axis=(0, 2, 3))
                    d_beta = cp.sum(gradients, axis=(0, 2, 3))
                    return d_inputs, d_gamma, d_beta
                else:
                    return d_inputs
            else:
                raise RuntimeError("Backward should not be called in eval mode")

        else:  # FC layer
            if self.train_mode:
                if self.affine:
                    d_x_hat = gradients * self.gamma
                else:
                    d_x_hat = gradients

                d_var = cp.sum(
                    d_x_hat * (self.inputs - self.current_mean) * -0.5 *
                    cp.power(self.current_var + self.epsilon, -1.5),
                    axis=0
                )

                d_mean = cp.sum(
                    d_x_hat * -1 / cp.sqrt(self.current_var + self.epsilon),
                    axis=0
                )

                d_inputs = d_x_hat / cp.sqrt(self.current_var + self.epsilon) + \
                           d_var * 2 * (self.inputs - self.current_mean) / gradients.shape[0] + \
                           d_mean / gradients.shape[0]

                if self.affine:
                    d_gamma = cp.sum(gradients * self.x_hat, axis=0)
                    d_beta = cp.sum(gradients, axis=0)
                    return d_inputs, d_gamma, d_beta
                else:
                    return d_inputs
            else:
                raise RuntimeError("Backward should not be called in eval mode")

    def set_mode(self, is_train):
        self.train_mode = is_train

    def get_state(self):
        return (self.gamma.copy() if self.affine else None,
                self.beta.copy() if self.affine else None,
                self.running_mean.copy(),
                self.running_var.copy())

    def set_state(self, state):
        if self.affine:
            self.gamma, self.beta, self.running_mean, self.running_var = state
        else:
            _, _, self.running_mean, self.running_var = state


class LeakyReLULayer:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.input_data = None

    def forward(self, input_data):
        self.input_data = input_data
        return cp.where(input_data > 0, input_data, self.alpha * input_data)

    def backward(self, output_gradient):
        d_input = cp.where(self.input_data > 0, 1, self.alpha)
        try:
            return output_gradient * d_input
        except ValueError:
            d_input = cp.broadcast_to(d_input, output_gradient.shape)
            return output_gradient * d_input


class DropoutLayer:
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.train_mode = True
        self.mask = None

    def forward(self, inputs):
        if not self.train_mode:
            return inputs

        self.mask = (cp.random.random(size=inputs.shape) > self.dropout_rate).astype(cp.float32)
        return inputs * self.mask / (1.0 - self.dropout_rate)

    def backward(self, gradients):
        if not self.train_mode:
            return gradients

        if isinstance(gradients, tuple):
            gradients = gradients[0]

        if self.mask is None:
            raise RuntimeError("Must call forward before backward")

        if gradients.shape != self.mask.shape:
            raise ValueError(f"Gradient shape {gradients.shape} doesn't match mask shape {self.mask.shape}")

        return gradients * self.mask / (1.0 - self.dropout_rate)

    def set_mode(self, is_train):
        self.train_mode = is_train


class MaxPoolingLayer:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, inputs):
        self.inputs = inputs
        B, C, H, W = inputs.shape
        PH = PW = self.pool_size
        S = self.stride

        out_h = (H - PH) // S + 1
        out_w = (W - PW) // S + 1

        input_reshaped = inputs.reshape(B, C, out_h, S, out_w, S)
        out = input_reshaped.max(axis=3).max(axis=4)
        self.max_mask = (inputs == cp.repeat(
            cp.repeat(out, S, axis=2), S, axis=3))
        return out

    def backward(self, dout):
        B, C, H, W = self.inputs.shape
        PH = PW = self.pool_size
        S = self.stride

        dX = cp.zeros_like(self.inputs)
        dout_repeated = cp.repeat(cp.repeat(dout, S, axis=2), S, axis=3)
        dX[self.max_mask] = dout_repeated[self.max_mask]
        return dX


class FlattenLayer:
    def __init__(self):
        pass

    def forward(self, inputs):
        self.inputs_shape = inputs.shape
        return inputs.reshape(inputs.shape[0], -1)

    def backward(self, gradients):
        return gradients.reshape(self.inputs_shape)


class DenseLayer:
    def __init__(self, input_size, output_size):
        scale = cp.sqrt(2.0 / input_size)
        self.weights = cp.random.normal(0, scale, (input_size, output_size))
        self.bias = cp.zeros(output_size)
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return cp.dot(x, self.weights) + self.bias

    def backward(self, dout):
        dx = cp.dot(dout, self.weights.T)
        self.dW = cp.dot(self.x.T, dout)
        self.db = cp.sum(dout, axis=0)
        return dx, self.dW, self.db


class Softmax:
    def __init__(self):
        pass

    def forward(self, inputs):
        self.inputs = inputs
        exp_values = cp.exp(inputs - cp.max(inputs, axis=-1, keepdims=True))
        probabilities = exp_values / cp.sum(exp_values, axis=-1, keepdims=True)
        self.outputs = probabilities
        return probabilities

    def backward(self, gradients):
        batch_size, num_classes = gradients.shape
        d_inputs = cp.empty_like(gradients)
        for i in range(batch_size):
            s = self.outputs[i].reshape(-1, 1)
            jacobian = cp.diagflat(s) - cp.dot(s, s.T)
            d_inputs[i] = cp.dot(jacobian, gradients[i])
        return d_inputs


class SGD:
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0, nesterov: bool = False):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocity = None

    def step(self, params, grads):
        if self.velocity is None:
            self.velocity = [cp.zeros_like(p) for p in params]

        for i, (param, grad) in enumerate(zip(params, grads)):
            # Update velocity
            self.velocity[i] = self.momentum * self.velocity[i] + grad

            if self.nesterov:
                # Nesterov accelerated gradient
                param -= self.learning_rate * (grad + self.momentum * self.velocity[i])
            else:
                # Standard momentum update
                param -= self.learning_rate * self.velocity[i]


class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 weight_decay=0.0001, amsgrad=False):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.m = None
        self.v = None
        if amsgrad:
            self.v_hat = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [cp.zeros_like(p) for p in params]
            self.v = [cp.zeros_like(p) for p in params]
            if self.amsgrad:
                self.v_hat = [cp.zeros_like(p) for p in params]

        self.t += 1
        for i in range(len(params)):
            # Apply weight decay
            params[i] *= (1 - self.learning_rate * self.weight_decay)

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i] ** 2)

            if self.amsgrad:
                self.v_hat[i] = cp.maximum(self.v_hat[i], self.v[i])
                v_hat = self.v_hat[i]
            else:
                v_hat = self.v[i]

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = v_hat / (1 - self.beta2 ** self.t)

            params[i] -= self.learning_rate * m_hat / (cp.sqrt(v_hat) + self.epsilon)


class AMPWrapper:
    """Automatic Mixed Precision Wrapper"""

    def __init__(self, model, optimizer, loss_scale=1024.0):
        self.model = model
        self.optimizer = optimizer
        self.loss_scale = loss_scale
        self._original_params = None

    def forward(self, x):
        # Convert input to float16
        x = x.astype(cp.float16)

        # Backup original parameters
        self._original_params = []
        for layer in self.model.layers:
            if hasattr(layer, 'weights'):
                self._original_params.append((layer.weights.copy(), layer.bias.copy()))
                layer.weights = layer.weights.astype(cp.float16)
                layer.bias = layer.bias.astype(cp.float16)

        # Forward pass
        output = self.model.forward(x)
        return output.astype(cp.float32)  # Convert output back to float32 for loss calculation

    def backward(self, grad_output):
        # Backward pass with float16
        grad_output = grad_output.astype(cp.float16)
        self.model.backward(grad_output)

        # Scale gradients before updating
        for layer in self.model.layers:
            if hasattr(layer, 'dW'):
                layer.dW = layer.dW.astype(cp.float32) / self.loss_scale
                layer.db = layer.db.astype(cp.float32) / self.loss_scale

        # Restore original parameters (float32)
        param_idx = 0
        for layer in self.model.layers:
            if hasattr(layer, 'weights'):
                layer.weights, layer.bias = self._original_params[param_idx]
                param_idx += 1

        # Update parameters
        params = []
        grads = []
        for layer in self.model.layers:
            if hasattr(layer, 'weights'):
                params.append(layer.weights)
                grads.append(layer.dW)
            if hasattr(layer, 'bias'):
                params.append(layer.bias)
                grads.append(layer.db)

        self.optimizer.step(params, grads)


class CrossEntropyLoss:
    def __init__(self, smoothing=0.1):
        self.smoothing = smoothing
        self.epsilon = 1e-12

    def forward(self, pred, target):
        pred = cp.clip(pred, self.epsilon, 1. - self.epsilon)
        self.pred = pred
        # Apply label smoothing
        self.target = target * (1 - self.smoothing) + self.smoothing / target.shape[1]
        loss = -cp.sum(self.target * cp.log(pred)) / pred.shape[0]
        return loss

    def backward(self):
        return (self.pred - self.target) / self.pred.shape[0]


class ResidualBlock:
    def __init__(self, in_channels, out_channels, stride=1):
        self.conv1 = ConvLayer(in_channels, out_channels, 3, stride, 1)
        self.bn1 = BatchNorm(out_channels)
        self.relu = LeakyReLULayer(0.1)
        self.conv2 = ConvLayer(out_channels, out_channels, 3, 1, 1)
        self.bn2 = BatchNorm(out_channels)

        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = SequentialLayer([
                ConvLayer(in_channels, out_channels, 1, stride, 0),
                BatchNorm(out_channels)
            ])
        else:
            self.shortcut = None

    def forward(self, x):
        self.identity = x  # 保存原始输入用于反向传播
        out = self.conv1.forward(x)
        out = self.bn1.forward(out)
        out = self.relu.forward(out)
        out = self.conv2.forward(out)
        out = self.bn2.forward(out)

        if self.shortcut is not None:
            self.identity = self.shortcut.forward(self.identity)

        out += self.identity
        self.output = self.relu.forward(out)  # 保存输出用于反向传播
        return self.output

    def backward(self, grad_output):
        # 反向传播经过最后的ReLU
        grad_output = self.relu.backward(grad_output)

        # 反向传播加法操作
        grad_conv = grad_output
        grad_shortcut = grad_output

        # 反向传播主路径
        grad_conv, _, _ = self.bn2.backward(grad_conv)
        grad_conv, _, _ = self.conv2.backward(grad_conv)
        grad_conv = self.relu.backward(grad_conv)
        grad_conv, _, _ = self.bn1.backward(grad_conv)
        grad_conv, _, _ = self.conv1.backward(grad_conv)

        # 反向传播shortcut路径
        if self.shortcut is not None:
            grad_shortcut = self.shortcut.backward(grad_shortcut)

        # 合并梯度
        grad_input = grad_conv + grad_shortcut
        return grad_input

    def set_mode(self, is_train):
        """设置训练/测试模式"""
        self.bn1.set_mode(is_train)
        self.bn2.set_mode(is_train)
        if self.shortcut is not None:
            for layer in self.shortcut.layers:
                if hasattr(layer, 'set_mode'):
                    layer.set_mode(is_train)


class GlobalAvgPoolLayer:
    def __init__(self):
        self.input_shape = None

    def forward(self, x):
        self.input_shape = x.shape
        return cp.mean(x, axis=(2, 3))  # Average over height and width

    def backward(self, grad_output):
        # 将梯度广播回原始尺寸
        grad_input = cp.zeros(self.input_shape, dtype=grad_output.dtype)
        elements = self.input_shape[2] * self.input_shape[3]
        grad_input += grad_output[:, :, cp.newaxis, cp.newaxis] / elements
        return grad_input

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