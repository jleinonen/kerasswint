from abc import ABC

import tensorflow as tf

from .swintbase import WindowAttentionBase, SwinTransformerBase


class WindowAttentionV2Base(WindowAttentionBase, ABC):
    def __init__(self, *args, **kwargs):
        self.rotation_invariant_bias = kwargs.pop(
            "rotation_invariant_bias", False
        )
        super().__init__(*args, **kwargs)
        self.logit_scale = tf.Variable(
            tf.ones((1,self.num_heads,1,1))*tf.math.log(10.0),
            trainable=True
        )

    @tf.function
    def windowed_attention(self, K, Q, V):
        max_scale = tf.math.log(100.0)
        logit_scale = tf.math.exp(tf.where(
            self.logit_scale > max_scale, max_scale, self.logit_scale
        ))
        K = tf.math.l2_normalize(K, axis=-1) * logit_scale
        Q = tf.math.l2_normalize(Q, axis=-1)
        return super().windowed_attention(K, Q, V)

    @tf.function
    def relative_position_bias(self):
        B = self.relative_position_bias_net(
            self.relative_position_bias_table
        )
        B = tf.transpose(B, (0,3,1,2))
        return B


class SwinTransformerV2Base(SwinTransformerBase, ABC):
    @tf.function
    def call(self, x):
        # Eq. 3 of Liu et al. (2021)
        x = self.norm1(self.wma(x)) + x
        x = self.norm2(self.mlp(x)) + x
        return x
