import jax
import numpy as np
import xarray as xr
import jax.numpy as jnp
from src.models import simulate
from tqdm import tqdm
import argparse


##############################################################################
#### Parsing arguments
##############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("a", help="hopf parameter", type=int)
parser.add_argument("frequency", help="natural oscillators frquecy", type=int)
parser.add_argument("amplitude", help="stimulus amplitude", type=float)
parser.add_argument("sweep", help="which parameter to sweep", type=str)
args = parser.parse_args()

##############################################################################
#### Simulation parameters
##############################################################################

Nareas = 2
ntrials = 200
fsamp = 1 / 1e-4

time = np.arange(-4, 4, 1 / fsamp)
beta = 1e-4
decim = 20
# time = time[::decim]

Npoints = len(time)

# C = np.array([[1, 0], [0, 1]]).T
C = np.array([[0, 1], [1, 0]]).T

# For the node receiving the input
a = args.a
f = args.frequency
I = args.amplitude
sweep = args.sweep

assert sweep in ["a", "f"]

# Stimulus
Iext = np.zeros((Nareas, Npoints))
Iext[0, (time >= 0) & (time <= 0.4)] = I
Amplitudes = np.linspace(0, 0.1, ntrials)
CS = Amplitudes[..., None, None] * Iext

simulate_loop = jax.vmap(
    simulate,
    in_axes=(None, None, None, None, None, None, None, 0, 0, None, None, None),
)

##############################################################################
#### Sweep parameter space
##############################################################################

nreps = 30

a_list = np.linspace(-10, 2, 15)
f_list = np.linspace(60, 10, 15)

time = time[::decim]


def sweep_a(a_list):

    def __loop(carry, a_):

        a, f = carry

        out = simulate_loop(
            C,
            1,
            jnp.array([f, f]),
            jnp.array([a, a_]),
            fsamp,
            beta,
            Npoints,
            CS,
            seeds,
            "cpu",
            decim,
            "both",
        )

        return carry, out

    _, out = jax.lax.scan(__loop, (a, f), a_list)

    out = xr.DataArray(
        np.stack(out),
        dims=("delta_a", "trials", "roi", "times"),
        coords=(
            a_list - a,
            Amplitudes,
            ["x", "y"],
            time,
        ),
    ).sel(times=slice(-2, 4))

    return out


def sweep_f(f_list):

    def __loop(carry, f_):

        a, f = carry

        out = simulate_loop(
            C,
            1,
            jnp.array([f, f_]),
            jnp.array([a, -7]),
            fsamp,
            beta,
            Npoints,
            CS,
            seeds,
            "cpu",
            decim,
            "both",
        )

        return carry, out

    _, out = jax.lax.scan(__loop, (a, f), f_list)

    out = xr.DataArray(
        np.stack(out),
        dims=("delta_f", "trials", "roi", "times"),
        coords=(
            f_list - f,
            Amplitudes,
            ["x", "y"],
            time,
        ),
    ).sel(times=slice(-2, 4))

    return out


"""
def sweep(a_list, f_list):

    params = np.array(np.meshgrid(a_list, f_list)).T.reshape(-1, 2)

    def __loop(carry, par):

        a, f = carry
        a_, f_ = par

        out = simulate_loop(
            C,
            1,
            jnp.array([f, f_]),
            jnp.array([a, a_]),
            fsamp,
            beta,
            Npoints,
            CS,
            seeds,
            "cpu",
            decim,
            "both",
        )

        return carry, out

    _, out = jax.lax.scan(__loop, (a, f), params)

    out = xr.DataArray(
        np.stack(out),
        dims=("params", "trials", "roi", "times"),
        coords=(
            range(len(params)),
            Amplitudes,
            ["x", "y"],
            time,
        ),
    ).sel(times=slice(-2, 4))

    return out
"""

ppe = []


for rep in tqdm(range(nreps)):

    seeds = np.random.randint(0, 10000, ntrials)

    if sweep == "a":
        data = sweep_a(a_list)
    else:
        data = sweep_f(f_list)

    data.to_netcdf(
        f"/home/INT/lima.v/Results/phase_encoding/2nodes/data_sweep_{sweep}_a_{a}_f_{f}_I_{I}_rep_{rep}.nc",
        auto_complex=True,
    )

    # data = sweep(a_list, f_list)

    # data.to_netcdf(
    #     f"/home/INT/lima.v/Results/phase_encoding/2nodes/data_a_f_I_{I}_rep_{rep}.nc",
    #     auto_complex=True,
    # )

    del data

    #    labels = np.tile(np.expand_dims(Amplitudes, 1), data.sizes["times"])
    #
    #    ppe_temp = []
    #
    #    for i, a_ in enumerate(a_list):
    #        z = data[i][:, 0] * np.conj(data[i][:, 1])
    #        A = np.abs(z).values
    #        dphi = np.unwrap(np.angle(z))
    #
    #        I_S_R1 = gcmi_nd_cc(A, labels, traxis=0)
    #        I_S_R2 = gcmi_nd_cc(dphi, labels, traxis=0)
    #
    #        ppe_temp += [np.trapz(I_S_R2 - I_S_R1, dx=np.diff(data.times.values)[0])]
    #
    #    ppe += [xr.DataArray(ppe_temp, dims=("delta_a"), coords=(a_list - a,))]
    #
    # ppe = xr.concat(ppe, "reps")
    # ppe.to_netcdf(f"/home/INT/lima.v/Results/phase_encoding/2nodes/ppe_I_{I}.nc")
