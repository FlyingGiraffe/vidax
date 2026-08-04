from typing import Union
import ml_dtypes
import numpy as np
import jax.numpy as jnp


def pt_tensor_to_numpy(pt_array: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
    """Materializes a torch.Tensor (any device/dtype) or numpy array as numpy.

    bfloat16 tensors are preserved as native bfloat16 (via `ml_dtypes`,
    already a JAX dependency) rather than upcast to float32 -- numpy itself
    has no bfloat16 type, and `torch.Tensor.numpy()` refuses bfloat16
    directly, so we reinterpret the raw bits instead of round-tripping
    through float32 (which would double memory for every T5/VAE checkpoint,
    both of which ship as native bfloat16).
    """
    if hasattr(pt_array, "detach"):
        pt_array = pt_array.detach().cpu()
        if hasattr(pt_array, "dtype") and str(pt_array.dtype) == "torch.bfloat16":
            import torch
            return pt_array.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
        return pt_array.numpy()
    return np.asarray(pt_array)


def convert_pt_tensor_to_jax(
    key: str,
    pt_array: Union[np.ndarray, "torch.Tensor"],
) -> jnp.ndarray:
    """Converts a single PyTorch state_dict tensor to its Flax-layout equivalent.

    Args:
        key: The parameter's PyTorch state_dict name (used to infer layout).
        pt_array: A torch.Tensor or numpy array.

    Returns:
        A JAX array with the layout Flax modules expect.
    """
    arr = pt_tensor_to_numpy(pt_array)

    # RMSNorm gamma (Wan2.1 VAE's `RMS_norm`): PyTorch keeps broadcastable
    # trailing singleton dims for channel-first layout, e.g. (dim, 1, 1) or
    # (dim, 1, 1, 1). Flax's RMSNorm scale is a flat (dim,) vector.
    if key.endswith(".gamma"):
        return jnp.array(arr.reshape(-1))

    # AdaLN modulation tensors, shape (1, 6, dim) or (1, 2, dim): no transpose.
    if "modulation" in key:
        return jnp.array(arr)

    if arr.ndim == 5:
        # PyTorch Conv3d: (Out, In, T, H, W) -> Flax Conv: (T, H, W, In, Out)
        arr = arr.transpose(2, 3, 4, 1, 0)
    elif arr.ndim == 4:
        # PyTorch Conv2d: (Out, In, H, W) -> Flax Conv: (H, W, In, Out)
        arr = arr.transpose(2, 3, 1, 0)
    elif arr.ndim == 2:
        # PyTorch Linear: (Out, In) -> Flax Dense kernel: (In, Out)
        arr = arr.T

    # 1D biases and norm scales are already in the right (and only) layout.
    return jnp.array(arr)
