from collections.abc import Iterable

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Activation, Dropout
from tensorflow.keras.layers import LayerNormalization

from .swintbase import WindowAttentionBase, SwinTransformerBase
from .swintbase import DualSwinTransformerBlockBase, PatchMergeBase
from .swintbase import PatchExpandBase
from .swintv2 import WindowAttentionV2Base, SwinTransformerV2Base


class WindowAttention3D(Layer):
    @staticmethod
    def spatial_dims():
        return 3

    def windowize(self, x):
        if any(self.shift):
            x = tf.roll(
                x,
                [-self.shift[0], -self.shift[1], -self.shift[2]],
                axis=(1,2,3)
            )
        
        (_, l, h, w, c) = x.shape
        ws = self.window_size
        wnum_l = l // ws[0]
        wnum_h = h // ws[1]
        wnum_w = w // ws[2]
        x = tf.reshape(
            x,
            (-1, wnum_l, ws[0], wnum_h, ws[1], wnum_w, ws[2], c)
        )
        x = tf.transpose(x, (0,1,3,5,2,4,6,7))
        wx = tf.reshape(x, shape=(-1, self.window_vol, c))
        return wx

    def dewindowize(self, wx, shape, channels):
        (l, h, w, c) = (shape[1], shape[2], shape[3], channels)
        ws = self.window_size
        wnum_l = l // ws[0]
        wnum_h = h // ws[1]
        wnum_w = w // ws[2]
        x = tf.reshape(
            wx,
            shape=(-1, wnum_l, wnum_h, wnum_w, ws[0], ws[1], ws[2], c),
        )
        x = tf.transpose(x, perm=(0,1,3,5,2,4,6,7))
        x = tf.reshape(x, shape=(-1, l, h, w, c))

        if any(self.shift):
            x = tf.roll(
                x,
                [self.shift[0], self.shift[1], self.shift[2]],
                axis=(1,2,3)
            )
        
        return x

    def init_attn_mask(self, shape, mask_value=-100):
        (depth, height, width) = (shape[1], shape[2], shape[3])
        if self.shift > 0:
            # calculate attention mask for SW-MSA
            img_mask = np.zeros((1, depth, height, width, 1))  # 1 H W 1
            def slices(ws, shift):
                return (
                    slice(0, -ws),
                    slice(-ws, -shift),
                    slice(-shift[0], None)
                )
            (d_slices, h_slices, w_slices) = (
                slices(*p) for p in zip(self.window_size, self.shift)
            )

            cnt = 0
            for d in d_slices:
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, d, h, w, :] = cnt
                        cnt += 1
            img_mask = tf.convert_to_tensor(img_mask,
                dtype=tf.keras.backend.floatx())

            mask_windows = self.windowize(img_mask)
            mask_windows = tf.reshape(
                mask_windows, shape=[-1, self.window_vol]
            )
            attn_mask = tf.expand_dims(mask_windows, axis=1) - \
                tf.expand_dims(mask_windows, axis=2)
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            self.attn_mask = tf.expand_dims(tf.expand_dims(attn_mask, axis=1), axis=0)
        else:
            self.attn_mask = None

    def init_relative_position_bias(self):
        ws = self.window_size
        num_window_elements = (2*ws[0] - 1) * (2*ws[1] - 1) * (2*ws[2] - 1)
        self.relative_position_bias_table = self.add_weight(
            shape=(num_window_elements, self.num_heads),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
        )
        coords_d = np.arange(ws[0])
        coords_h = np.arange(ws[1])
        coords_w = np.arange(ws[2])
        coords_matrix = np.meshgrid(coords_d, coords_h, coords_w, indexing="ij")
        coords = np.stack(coords_matrix)
        coords_flatten = coords.reshape(3, -1)
        relative_coords = coords_flatten[:, :, None] - \
            coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += ws[0] - 1
        relative_coords[:, :, 1] += ws[1] - 1
        relative_coords[:, :, 2] += ws[2] - 1
        relative_coords[:, :, 0] *= (2*ws[1] - 1) * (2*ws[2] - 1)
        relative_coords[:, :, 1] *= 2*ws[2] - 1
        relative_position_index = relative_coords.sum(axis=-1)
        relative_position_index = tf.reshape(
            relative_position_index, shape=(-1,)
        )
        self.relative_position_index = tf.convert_to_tensor(
            relative_position_index
        )


def WindowAttention3DV2(WindowAttentionV2Base,WindowAttention3D)
    def init_relative_position_bias(self):
        ind = tf.range(self.window_vol)
        i = ind // (self.window_size * self.window_size)
        j = (ind // self.window_size) % self.window_size
        k = ind % self.window_size
        ijk = tf.stack([i,j,k], axis=-1)
        rel_pos = ijk[:,None,:] - ijk[None,:,:]
        rel_pos = tf.cast(dist, tf.keras.backend.floatx())
        if self.rotation_invariant_bias:
            abs_dist = tf.math.sqrt(
                tf.math.square(rel_pos[:,:,1]) + 
                tf.math.square(rel_pos[:,:,2])
            )
            rel_pos = tf.stack([rel_pos[:,:,0], abs_dist], axis=-1)
        rel_pos = rel_pos * 8 / (self.window_size - 1)
        rel_pos = tf.sign(rel_pos) * \
            tf.math.log(tf.abs(rel_pos)+1) / tf.math.log(8)
        rel_pos = tf.expand_dims(rel_pos, axis=0)
        self.relative_position_bias_table = rel_pos

        self.relative_position_bias_net = tf.keras.Sequential([
            Dense(512, activation="swish"),
            Dense(self.num_heads)
        ])


class SwinTransformer3D(SwinTransformerBase):
    @staticmethod
    def attention_class():
        return WindowAttention3D


class SwinTransformer3DV2(SwinTransformerV2Base):
    @staticmethod
    def attention_class():
        return WindowAttention3DV2


class DualSwinTransformerBlock3D(DualSwinTransformerBlockBase):
    @staticmethod
    def swint_class():
        return SwinTransformer3D


class DualSwinTransformerBlock3DV2(DualSwinTransformerBlockBase):
    @staticmethod
    def swint_class():
        return SwinTransformer3DV2


class PatchMerge3D(PatchMergeBase):
    @staticmethod
    def spatial_dims():
        return 3


class PatchExpand3D(PatchExpandBase):
    @staticmethod
    def spatial_dims():
        return 3

