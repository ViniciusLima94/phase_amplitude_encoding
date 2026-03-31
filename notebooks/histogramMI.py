from jax import custom_derivatives, dtypes, lax
from jax._src.lax.lax import _const as _lax_const
from jax._src.numpy.util import promote_args_inexact, promote_dtypes_inexact
from jax.typing import ArrayLike
import jax
import jax.numpy as jnp

import numpy as np

def panzeri_treves_correction(I_obs, R, S, N):
    bias = ((R - 1) * (S - 1)) / (2 * N * np.log(2))
    return I_obs - bias


def digitize(data, N, mode="equiprobable"):
    """
    Coarse-grain continuous data into N equiprobable bins (quantile-based).

    Each sample is replaced by the bin's representative (mean) value.

    Parameters:
        data (array-like): input float data (any shape)
        N (int): number of bins

    Returns:
        np.ndarray: data replaced by bin representative values
        np.ndarray: bin edges used
    """
    assert mode in ["linear", "equiprobable"]

    data = jnp.asarray(data)

    # Compute quantile-based bin edges
    quantiles = jnp.linspace(0, 1, N + 1)
    if mode == "equiprobable":
        bin_edges = jnp.quantile(data, quantiles, axis=1).T
    else:
        bin_edges = jnp.linspace(data.min(axis=1), data.max(axis=1), N).T

    bin_idx = jax.vmap(jnp.digitize, in_axes=(0, 0, None))(
        data, bin_edges, True
    )

    return bin_idx, bin_edges


def xlogy(x: ArrayLike, y: ArrayLike):
    """Compute x*log(y), returning 0 for x=0.

    JAX implementation of :obj:`scipy.special.xlogy`.

    This is defined to return zero when :math:`(x, y) = (0, 0)`, with a custom
    derivative rule so that automatic differentiation is well-defined at this point.

    Args:
      x: arraylike, real-valued.
      y: arraylike, real-valued.

    Returns:
      array containing xlogy values.

    See also:
      :func:`jax.scipy.special.xlog1py`
    """
    # Note: xlogy(0, 0) should return 0 according to the function documentation.
    x, y = promote_args_inexact("xlogy", x, y)
    x_ok = x != 0.0
    return jnp.where(x_ok, lax.mul(x, lax.log(y)), jnp.zeros_like(x))


def entrBin(x: ArrayLike, deltas: ArrayLike):
    r"""The entropy function

  JAX implementation of :obj:`scipy.special.entr`.

  .. math::

     \mathrm{entr}(x) = \begin{cases}
       -x\log(x) & x > 0 \\
       0 & x = 0\\
       -\infty & \mathrm{otherwise}
     \end{cases}

  Args:
    x: arraylike, real-valued.

  Returns:
    array containing entropy values.

  See also:
    - :func:`jax.scipy.special.kl_div`
    - :func:`jax.scipy.special.rel_entr`
  """
    (x,) = promote_args_inexact("entr", x)
    if dtypes.issubdtype(x.dtype, jnp.complexfloating):
        raise ValueError("entr does not support complex-valued inputs.")
    return lax.select(
        lax.lt(x, _lax_const(x, 0)),
        lax.full_like(x, -jnp.inf),
        lax.neg(xlogy(x, x / deltas)),
    )


def entropy_discrete(x, n_bins, mode = "equiprobable"):
    N = x.shape[1]

    x, bin_edges = digitize(x, n_bins)

    deltas = jnp.atleast_2d( jnp.diff(bin_edges, axis=1) )

    # Entropy
    uniques, counts = jnp.unique(x, return_counts=True, axis=1, size=N, fill_value=0)
    probs = counts / N

    delta_per_state = jnp.prod(
        deltas[jnp.arange(deltas.shape[0])[:, None], uniques], axis=0
    )
    return entrBin(probs, delta_per_state).sum() / jnp.log(2)


def mutualInformationBin(x, y, n_bins, mode):

    Hx = entropy_discrete(x, n_bins, mode)
    Hy = entropy_discrete(y, n_bins, mode)
    Hxy = entropy_discrete(
        jnp.concatenate((x, y), axis=0),
        n_bins,
        mode,
    )
    return Hx + Hy - Hxy


mutualInformationBin_jax = jax.vmap(mutualInformationBin, in_axes=(2, 2, None, None))