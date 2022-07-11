from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Activation, Dropout
from tensorflow.keras.layers import LayerNormalization

class WindowAttentionBase(Layer, ABC):
    def __init__(
        self,
        dim_in,
        self_attention=True,
        num_heads=4,
        window_size=4,
        shift=0,
        dropout=0,
        dim_out=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.self_attention = self_attention
        if isinstance(window_size, Iterable):
            self.window_size = tuple(window_size)
            self.shift = tuple(shift)
        else:
            self.window_size = (window_size,) * self.spatial_dims()
            self.shift = (shift,) * self.spatial_dims()
        self.window_vol = int(np.prod(self.window_size))

        self.num_heads = num_heads
        if dim_in % num_heads:
            raise ValueError("dim_in must be divisible by num_heads")
        self.dim_head = dim_in // num_heads
        if dim_out is None:
            dim_out = dim_in
        self.dim_out = dim_out
        self.attn_scale = 1.0 / tf.math.sqrt(
            tf.cast(self.dim_head, tf.keras.backend.floatx())
        )

        self.num_KVQ_conv = 3 if self_attention else 2
        self.dim_KVQ_conv = self.num_KVQ_conv * dim_in
        self.KVQ = Dense(self.dim_KVQ_conv)
        self.dropout = Dropout(dropout)

        self.proj = Dense(self.dim_out)
        self.init_relative_position_bias()

    @staticmethod
    @abstractmethod
    def spatial_dims():
        pass

    def build(self, input_shape):
        if not self.self_attention:
            input_shape = input_shape[0]
        self.init_attn_mask(input_shape)

    @abstractmethod
    def windowize(self, x):
        pass

    @abstractmethod
    def dewindowize(self, wx, shape, channels):
        pass

    @tf.function
    def get_KVQ(self, U):
        KVQ = self.KVQ(U)
        shape = [
            -1,
            self.num_heads,
            self.window_vol,
            self.dim_head,
            self.num_KVQ_conv
        ]
        KVQ = tf.reshape(KVQ, shape)
        return tf.unstack(KVQ, axis=-1)

    @abstractmethod
    def init_attn_mask(self, shape, mask_value=-100.0):
        pass

    @abstractmethod
    def init_relative_position_bias(self):
        pass

    @tf.function
    def relative_position_bias(self):
        B = tf.gather(
            self.relative_position_bias_table,
            self.relative_position_index
        )
        B = tf.reshape(
            B,
            shape=(1, self.window_vol, self.window_vol, -1)
        )
        return tf.transpose(B, (0,3,1,2))

    @tf.function
    def apply_attn_mask(self, attn):
        if self.attn_mask is None:
            return attn
        else:
            wvol = self.window_vol
            nw = self.attn_mask.shape[1]
            attn = tf.reshape(attn, (-1, nw, self.num_heads, wvol, wvol))
            attn = attn + self.attn_mask
            return tf.reshape(attn, (-1, self.num_heads, wvol, wvol))

    @tf.function
    def windowed_attention(self, K, V, Q):
        # compute dot product attention window-wise
        attn = tf.matmul(Q, K, transpose_b=True) # Q * K^T
        attn = attn * self.attn_scale
        attn = self.apply_attn_mask(attn)
        attn = attn + self.relative_position_bias()
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.dropout(attn)
        
        # apply attention
        attn = tf.matmul(attn, V) # attn * V
        
        attn = tf.transpose(attn, (0,2,1,3)) # swap head and window dims
        # concatenate heads and project
        shape = tf.concat([tf.shape(attn)[:2], [self.dim_out]], axis=0)
        attn = tf.reshape(attn, shape)
        attn = self.proj(attn)
        return self.dropout(attn)

    @tf.function
    def call(self, inputs):
        if self.self_attention:
            X = inputs
        else:
            (X, Q) = inputs
            Q = self.windowize(Q)
        X_shape = X.shape
        X = self.windowize(X)

        if self.self_attention:
            (K, V, Q) = self.get_KVQ(X)
        else:
            (K, V) = self.get_KVQ(X)
            
        Y = self.windowed_attention(K, V, Q)
        Y = self.dewindowize(Y, X_shape, self.dim_out)
        return Y


class SwinTransformerBase(Layer, ABC):
    def __init__(
        self,
        dim,
        num_heads=8,
        window_size=4,
        shift=0,
        dropout=0,
        norm='layer',
        norm_groups=8,
        mlp=None,
        activation=tf.keras.activations.gelu,
        mlp_ratio=4,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.dim = dim
        self.dropout = dropout
        self.activation = activation

        # attention layer
        attn_kwargs = {
            "self_attention": True,
            "num_heads": num_heads,
            "window_size": window_size,
            "dropout": dropout,
            "shift": shift
        }
        self.wma = self.attention_class()(dim, **attn_kwargs)
        
        # feedforward MLP network layers
        self.mlp_ratio = mlp_ratio
        create_mlp = mlp if mlp is not None else self.create_mlp
        self.mlp = create_mlp()
        
        create_norm = {
            None: lambda: (lambda x: x),
            "layer": lambda: LayerNormalization(scale=False)
        }[norm]
        (self.norm1, self.norm2) = (create_norm() for _ in range(2))

    @staticmethod
    @abstractmethod
    def attention_class():
        pass

    def create_mlp(self):
        return tf.keras.Sequential(
            [
                Dense(self.dim*self.mlp_ratio, activation=self.activation),
                Dropout(self.dropout),
                Dense(self.dim),
                Dropout(self.dropout),
            ]
        )

    @tf.function
    def call(self, x):
        # Eq. 3 of Liu et al. (2021)
        x = self.wma(self.norm1(x)) + x
        x = self.mlp(self.norm2(x)) + x
        return x


class DualSwinTransformerBlockBase(Layer, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(
            name=kwargs.pop("name", None),
            trainable=kwargs.get("trainable", True),
            dtype=kwargs.get("dtype")
        )
        self.window_size = kwargs.get("window_size", 4)
        self.shift = kwargs.pop("shift", self.window_size // 2)
        self.swint1 = self.swint_class()(*args, **kwargs)
        self.swint2 = self.swint_class()(*args, shift=self.shift, **kwargs)

    @staticmethod
    @abstractmethod
    def swint_class():
        pass

    def call(self, x):
        x = self.swint1(x)
        x = self.swint2(x)
        return x


class PatchMergeBase(Layer, ABC):
    def __init__(self, channels=None, size=2, activation=None, 
        use_bias=False, **kwargs):

        super().__init__(**kwargs)
        self.channels = channels
        if not isinstance(size, Iterable):
            size = (size,) * self.spatial_dims()
        self.size = size
        self.activation = activation
        self.use_bias = use_bias

    @staticmethod
    @abstractmethod
    def spatial_dims():
        pass

    def build(self, input_shape):
        if self.channels is None:
            self.channels = input_shape[-1] * 2
        self.reduction = Dense(self.channels, activation=self.activation,
            use_bias=self.use_bias)

    @tf.function
    def call(self, x):
        p = []
        s = self.size
        ndim = self.spatial_dims()

        for ind in product(range(d) for d in s)):
            slices = (slice(None),) + \
                tuple(slice(ind[i], None, s[i]) for i in range(ndim)) + \
                (slice(None),)
            p = p + p[slices]

        p = tf.concat(p, axis=-1)
        return self.reduction(p)


class PatchExpandBase(Layer):
    def __init__(self, channels=None, size=2, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        if not isinstance(size, Iterable):
            size = (size,) * self.spatial_dims()
        self.vol = int(np.prod(size))
        self.size = size

    @staticmethod
    @abstractmethod
    def spatial_dims():
        pass

    def build(self, input_shape):
        if self.channels is None:
            self.channels = input_shape[-1] // 2
        self.expansion = Dense(self.vol * self.channels, use_bias=False)

    @tf.function
    def call(self, x):
        p = self.expansion(x)
        shape = tf.concat(
            [tf.shape(p)[:-1], self.size, [self.channels]],
            axis=0
        )
        p = tf.reshape(p, shape)
        channel_dim = self.spatial_dims()*2+1
        dims = range(1,channel_dim)
        transpose_dims = (0,) + dims[::2] + dims[1::2] + (channel_dim,)
        p = tf.transpose(p, transpose_dims)
        s = tf.shape(p)
        shape = tf.stack(
            [
                s[0] + 
                [s[i]*s[i+1] for i in range(1,channel_dim,2)] +
                s[channel_dim]
            ],
            axis=0
        )
        return tf.reshape(p, shape)

