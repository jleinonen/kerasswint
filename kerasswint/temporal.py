import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Activation, Dropout
from tensorflow.keras.layers import LayerNormalization


class TemporalMultiheadAttention(Layer):
    def __init__(
        self,
        dim_in,
        num_heads=4,
        dropout=0,
        dim_out=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.num_heads = num_heads
        if dim_in % num_heads:
            raise ValueError("dim_in must be divisible by num_heads")
        self.dim_head = dim_in // num_heads
        if dim_out is None:
            dim_out = dim_in
        self.dim_out = dim_out

        self.dim_KVQ_conv = 2 * dim_in
        self.KV = Dense(self.dim_KVQ_conv)
        self.Q = Dense(dim_in)

        self.dropout = Dropout(dropout)

        self.proj = Dense(self.dim_out)

    #@tf.function
    def get_KVQ(self, X, Y):
        KV = self.KV(X)
        shape = tf.concat(
            [
                tf.shape(X)[:4],
                [self.num_heads, self.dim_head, 2]
            ],
            axis=0
        )
        KV = tf.reshape(KV, shape)
        (K, V) = tf.unstack(KV, axis=-1)
        
        Q = self.Q(Y)
        shape = tf.concat(
            [
                tf.shape(Y)[:4],
                [self.num_heads, self.dim_head]
            ],
            axis=0
        )
        Q = tf.reshape(Q, shape)

        return (K, V, Q)

    #@tf.function
    def temporal_attention(self, K, V, Q):
        # swap time coordinate to dimension -2 (-1 for K)
        K = tf.transpose(K, (0,2,3,4,5,1))
        V = tf.transpose(V, (0,2,3,4,1,5))
        Q = tf.transpose(Q, (0,2,3,4,1,5))

        # compute attention in time coordinates
        attn = tf.matmul(Q, K) # Q * K^T
        print(attn.shape)
        attn_scale = 1.0 / tf.math.sqrt(tf.cast(K.shape[-1], attn.dtype))
        attn = attn * attn_scale
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.dropout(attn)
        
        # apply attention
        attn = tf.matmul(attn, V) # attn * V
        
        attn = tf.transpose(attn, (0,4,1,2,3,5)) # swap time dimension back
        # concatenate heads and project
        shape = tf.concat([tf.shape(attn)[:4], [self.dim_out]], axis=0)
        attn = tf.reshape(attn, shape)
        attn = self.proj(attn)
        return self.dropout(attn)

    #@tf.function
    def call(self, inputs):
        (X, Y) = inputs
        (K, V, Q) = self.get_KVQ(X, Y)
        Y = self.temporal_attention(K, V, Q)
        return Y


class AddPosEncoding(Layer):
    def __init__(self, d_model, timesteps, scale=10000.0):
        super().__init__()

        float_dtype = tf.keras.backend.floatx()
        timesteps = tf.constant(timesteps, dtype=float_dtype)
        timesteps = tf.expand_dims(timesteps, axis=-1)

        i = tf.range(d_model//2, dtype=float_dtype)
        arg = tf.math.pow(scale, 2*i/d_model)
        arg = tf.expand_dims(arg, axis=0)
        arg = timesteps / arg

        s = tf.math.sin(arg)
        c = tf.math.cos(arg)
        self.encoding = tf.concat([s,c], axis=-1)
        for dim in [0, 2, 3]:
            self.encoding = tf.expand_dims(self.encoding, axis=dim)

    def call(self, x):
        return x + self.encoding


class TemporalTransformer(Layer):
    def __init__(
        self,
        dim,
        num_heads=8,
        dropout=0,
        norm='layer',
        norm_groups=8,
        mlp=None,
        activation=tf.keras.activations.swish,
        mlp_ratio=4,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.dim = dim
        self.dropout = dropout
        self.activation = activation

        # attention layer
        shift = window_size // 2
        attn_kwargs = {
            "self_attention": True,
            "num_heads": num_heads,
            "dropout": dropout,
        }
        self.tma = TemporalMultiheadAttention(dim, **attn_kwargs)
        
        # feedforward MLP network layers
        self.mlp_ratio = mlp_ratio
        create_mlp = mlp if mlp is not None else self.create_mlp
        self.mlp = create_mlp()
        
        create_norm = {
            None: lambda: (lambda x: x),
            "layer": lambda: LayerNormalization(scale=False)
        }[norm]
        (self.norm1, self.norm2) = (create_norm() for _ in range(2))

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
        x = self.tma(self.norm1(x)) + x
        x = self.mlp(self.norm2(x)) + x
        return x

