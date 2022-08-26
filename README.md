## Swin Transformer implementations

This is a set of unified Keras/TensorFlow implementations of:
* Swin Transformer (Liu et al. 2021, https://arxiv.org/abs/2103.14030)
* Swin Transformer V2 (Liu et al. 2022, https://arxiv.org/abs/2111.09883)
* 3D (Video) Swin Transformer (Liu et al. 2021, https://arxiv.org/abs/2106.13230)

There is also an implementation of a global transformer along the temporal axis.

I think the 2D Swin Transformer and V2 are working as intended. The 3D and temporal transformers are less well tested, but the 3D version is implemented such that it shares a lot of the code with the 2D versions. Contributions are welcome.

## Installation

Clone the repository:
```bash
git clone https://github.com/jleinonen/kerasswint
```

Install the package:
```bash
pip install .
```
Or if you want to edit the files while installed:
```bash
pip install -e .
```

## Usage

Documentation is scarce at the moment. See the demo at `scripts/mnist_demo.py` for an example.

The classes of the Swin Transformer implementation are as follows:
* `swint.WindowAttention2D`: windowed multi-head attention (WMA), with optional shift, as in the original Swin Transformer paper
* `swint.WindowAttention2DV2`: Swin Transformer V2 version of the above
* `swint.SwinTransformer2D`: WMA + MLP transformer block
* `swint.SwinTransformer2DV2`: Swin Transformer V2 version of the above
* `swint.DualSwinTransformerBlock2D`: Block with two Swin Transformers with the second shifted (as used in the paper)
* `swint.DualSwinTransformerBlock2DV2`: Swin Transformer V2 version of the above

The equivalent 3D implementations are in `swint3d`, just replace `2D` with `3D` in the above list.
