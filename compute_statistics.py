import os
import xarray as xr
import numpy as np
import argparse
from hoi.metrics import SynergyMMI, RedundancyMMI
from frites.workflow import WfStats
from frites.utils import parallel_func


##############################################################################
# PARSE ARGUMENTS
##############################################################################


parser = argparse.ArgumentParser(description="Generate Hopf simulaitons")
parser.add_argument("--f", type=float, default=40, help="nodes oscillating frequency")
parser.add_argument("--a", type=float, default=-5.0, help="nodes hopf parameter")
parser.add_argument("--g", type=float, default=10, help="global coupling scaling")
parser.add_argument("--eta", type=float, default=1, help="exitatory scaling")
parser.add_argument("--seed", type=int, default=0, help="seed used")
args = parser.parse_args()


_SAVE = os.path.join("Results", "statistics")

if not os.path.exists(_SAVE):

    os.makedirs(_SAVE)

##############################################################################
# FUNCTIONS
##############################################################################


def fit_model(model_class, x, y, minsize=2, maxsize=2):
    """
    Fit a single MMI model for given data and labels.
    """
    model = model_class(x, y, verbose=False)
    hoi = model.fit(minsize=minsize, maxsize=maxsize)
    return hoi


def SRMMI_for_subject(data, y, n_perm, hoi_type):
    """
    Compute synergy or redundancy via multivariate mutual information (MMI)
    with permutation testing and cluster-based stats.

    Parameters
    ----------
    data : xarray.DataArray
        dims (subjects, roi, times)
    sbj: int
        subject number
    y : array-like
        trial labels, shape (n_trials,)
    n_perm : int
        number of label permutations
    hoi_type : {'red', 'syn'}

    Returns
    -------
    hoi_mean : ndarray, shape (n_links,)
    p_values : ndarray, shape (n_links, n_times)
    test_vals : ndarray, shape (n_links, n_times)
    links : ndarray of str, shape (n_links,)
    times : ndarray, shape (n_times,)
    """

    # HOI model
    if hoi_type == "red":
        model_class = RedundancyMMI
    elif hoi_type == "syn":
        model_class = SynergyMMI

    x = data.values

    # Compute true HOI
    model_hoi = model_class(x, y)
    _hoi = model_hoi.fit(minsize=2, maxsize=2)

    # define the function to compute in parallel
    parallel, p_fun = parallel_func(
        fit_model, n_jobs=10, verbose=False, total=n_perm, prefer="threads"
    )
    # Compute the single trial coherence
    perm = parallel(
        p_fun(model_class, x, np.random.permutation(y), minsize=2, maxsize=2)
        for _ in range(n_perm)
    )
    # Stack to form a (n_permutations, n_combinations, n_variables) array
    surrogates = np.stack(perm, axis=0)

    hoi_s = [hoi_[None] for hoi_ in _hoi]
    hoi_p_s = [hoi_[:, None] for hoi_ in surrogates.swapaxes(0, 1)]

    tail = 1
    inference = "ffx"
    mcp = "cluster"
    wf = WfStats()
    pv, tv = wf.fit(hoi_s, hoi_p_s, inference=inference, mcp=mcp, tail=tail)
    # Mean across subjects
    links = [f"{s}-{t}" for s, t in model_hoi.multiplets]  # noqa
    hoi_m = np.mean(np.stack(hoi_s, axis=0), axis=1)
    hoi_m = xr.DataArray(
        hoi_m,
        dims=("roi", "times"),
        coords=(
            links,
            data.times.values,
        ),
    )

    pv = xr.DataArray(
        pv.T,
        dims=("roi", "times"),
        coords=(
            links,
            data.times.values,
        ),
    )

    return hoi_m, pv, tv


def get_file_name(f, a, g, eta, seed):
    return f"data_f_{f}_a_{a}_g_{g}_eta_{eta}_seed_{seed}.nc"


if __name__ == "__main__":

    # Node and network parameters
    f = args.f
    a = args.a
    g = args.g
    eta = args.eta
    seed = args.seed

    # Number of permutations
    n_perm = 500

    data = xr.open_dataarray(
        os.path.join("Results", "dynamics", get_file_name(f, a, g, eta, seed))
    )
    y = data.attrs["y"]

    syn, p_syn, t_syn = SRMMI_for_subject(data, y, n_perm, "syn")
    red, p_red, t_red = SRMMI_for_subject(data, y, n_perm, "red")

    syn.to_netcdf(os.path.join(_SAVE, f"syn_{f}_a_{a}_g_{g}_eta_{eta}_seed_{seed}.nc"))
    p_syn.to_netcdf(
        os.path.join(_SAVE, f"pvalue_syn_{f}_a_{a}_g_{g}_eta_{eta}_seed_{seed}.nc")
    )

    red.to_netcdf(os.path.join(_SAVE, f"red_{f}_a_{a}_g_{g}_eta_{eta}_seed_{seed}.nc"))
    p_red.to_netcdf(
        os.path.join(_SAVE, f"pvalue_red_{f}_a_{a}_g_{g}_eta_{eta}_seed_{seed}.nc")
    )
