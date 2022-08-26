import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

from .swintbase import WindowAttentionBase, SwinTransformerBase
from .swintbase import DualSwinTransformerBlockBase, PatchMergeBase
from .swintbase import PatchExpandBase
from .swintv2 import WindowAttentionV2Base, SwinTransformerV2Base


class WindowAttention2D(WindowAttentionBase):
    @staticmethod
    def spatial_dims():
        return 2

    def windowize(self, x):
        if any(self.shift):
            x = tf.roll(x, [-self.shift[0], -self.shift[1]], axis=(1,2))
        
        (_, h, w, c) = x.shape
        ws = self.window_size
        patch_num_h = h // ws[0]
        patch_num_w = w // ws[1]
        x = tf.reshape(
            x, shape=(-1, patch_num_h, ws[0], patch_num_w, ws[1], c)
        )
        x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
        wx = tf.reshape(x, shape=(-1, self.window_vol, c))
        return wx

    def dewindowize(self, wx, shape, channels):
        (h, w, c) = (shape[1], shape[2], channels)
        ws = self.window_size
        patch_num_h = h // ws[0]
        patch_num_w = w // ws[1]
        x = tf.reshape(
            wx,
            shape=(-1, patch_num_h, patch_num_w, ws[0], ws[1], c),
        )
        x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
        x = tf.reshape(x, shape=(-1, h, w, c))

        if any(self.shift):
            x = tf.roll(x, [self.shift[0], self.shift[1]], axis=(1,2))
        
        return x

    def init_attn_mask(self, shape, mask_value=-100.0):
        (height, width) = (shape[1], shape[2])
        if any(self.shift):
            # calculate attention mask for SW-MSA
            img_mask = np.zeros((1, height, width, 1))
            h_slices = (slice(0, -self.window_size[0]),
                        slice(-self.window_size[0], -self.shift[0]),
                        slice(-self.shift[0], None))
            w_slices = (slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], -self.shift[1]),
                        slice(-self.shift[1], None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            img_mask = tf.convert_to_tensor(img_mask,
                dtype=tf.keras.backend.floatx())

            mask_windows = self.windowize(img_mask)
            mask_windows = tf.reshape(
                mask_windows, shape=[-1, self.window_vol]
            )
            attn_mask = tf.expand_dims(mask_windows, axis=1) - \
                tf.expand_dims(mask_windows, axis=2)
            attn_mask = tf.where(attn_mask != 0, mask_value, attn_mask)
            self.attn_mask = tf.expand_dims(tf.expand_dims(attn_mask, axis=1), axis=0)
        else:
            self.attn_mask = None

    def init_relative_position_bias(self):
        ws = self.window_size
        num_window_elements = (2*ws[0] - 1) * (2*ws[1] - 1)
        self.relative_position_bias_table = self.add_weight(
            shape=(num_window_elements, self.num_heads),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
        )
        coords_h = np.arange(ws[0])
        coords_w = np.arange(ws[1])
        coords_matrix = np.meshgrid(coords_h, coords_w, indexing="ij")
        coords = np.stack(coords_matrix)
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - \
            coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += ws[0] - 1
        relative_coords[:, :, 1] += ws[1] - 1
        relative_coords[:, :, 0] *= 2*ws[1] - 1
        relative_position_index = relative_coords.sum(-1)
        relative_position_index = tf.reshape(
            relative_position_index, shape=(-1,)
        )
        self.relative_position_index = tf.convert_to_tensor(
            relative_position_index
        )


class WindowAttention2DV2(WindowAttentionV2Base,WindowAttention2D):
    def init_relative_position_bias(self):
        ind = tf.range(self.window_vol)
        i = ind // self.window_size[1]
        j = ind % self.window_size[1]
        ij = tf.stack([i,j], axis=-1)
        rel_pos = ij[:,None,:] - ij[None,:,:]
        rel_pos = tf.cast(rel_pos, tf.keras.backend.floatx())
        if self.rotation_invariant_bias:
            rel_pos = tf.math.sqrt(
                tf.math.square(rel_pos[:,:,0:1]) + 
                tf.math.square(rel_pos[:,:,1:2])
            )
        rel_pos = rel_pos * 8 / (max(self.window_size) - 1)
        rel_pos = tf.sign(rel_pos) * \
            tf.math.log(tf.abs(rel_pos)+1) / tf.math.log(8.0)
        rel_pos = tf.expand_dims(rel_pos, axis=0)
        self.relative_position_bias_table = rel_pos

        self.relative_position_bias_net = tf.keras.Sequential([
            Dense(512, activation="relu"),
            Dense(self.num_heads)
        ])


class SwinTransformer2D(SwinTransformerBase):
    @staticmethod
    def attention_class():
        return WindowAttention2D


class SwinTransformer2DV2(SwinTransformerV2Base):
    @staticmethod
    def attention_class():
        return WindowAttention2DV2


class DualSwinTransformerBlock2D(DualSwinTransformerBlockBase):
    @staticmethod
    def swint_class():
        return SwinTransformer2D


class DualSwinTransformerBlock2DV2(DualSwinTransformerBlockBase):
    @staticmethod
    def swint_class():
        return SwinTransformer2DV2


class PatchMerge2D(PatchMergeBase):
    @staticmethod
    def spatial_dims():
        return 2


class PatchExpand2D(PatchExpandBase):
    @staticmethod
    def spatial_dims():
        return 2


class PatchEmbedding2D(PatchMerge2D):
    def __init__(self, channels=96, size=4, pos_embed=False, **kwargs):
        super().__init__(channels=channels, size=size, **kwargs)
        self.pos_embed = pos_embed

    def build(self, input_shape):
        super().build(input_shape)
        if self.pos_embed:
            self.num_patch_h = input_shape[1] // self.size[0]
            self.num_patch_v = input_shape[2] // self.size[1]
            num_patch = self.num_patch_h * self.num_patch_v
            self.pos_embedding = tf.keras.layers.Embedding(
                num_patch, self.channels
            )
            self.pos = tf.range(num_patch)

    @tf.function
    def call(self, x):
        x = super().call(x)
        if self.pos_embed:
            embed = self.pos_embedding(self.pos)
            s = tf.shape(embed)
            embed = tf.reshape(
                embed, 
                [1, self.num_patch_h, self.num_patch_v, self.channels]
            )
            x = x + embed
        return x
