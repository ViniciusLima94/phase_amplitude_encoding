#### Simulate Hopf dynamic of top of a specified network structure
import jax
import numpy as np
import argparse
import os

import xarray as xr


from src.models import simulate
from tqdm import tqdm  # noqa
from jax_tqdm import scan_tqdm  # noqa
from functools import partial  # noqa

import random

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # add this
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

##############################################################################
# PARSE ARGUMENTS
##############################################################################

parser = argparse.ArgumentParser(description="Generate Hopf simulaitons")
parser.add_argument("--f", type=float, default=1, help="nodes oscillating frequency")
parser.add_argument("--a", type=float, default=-3.0, help="nodes hopf parameter")
parser.add_argument("--g", type=float, default=1, help="global coupling scaling")
parser.add_argument("--eta", type=float, default=1, help="exitatory scaling")
parser.add_argument("--ntrials", type=int, default=500, help="number of trials")
parser.add_argument("--dt", type=float, default=1e-4, help="integration time step")
parser.add_argument("--seed", type=int, default=0, help="seed used")
args = parser.parse_args()


seed = args.seed  # seed for random number generation
random.seed(seed)
np.random.seed(seed)

jax.config.update("jax_platform_name", "cpu")


# Path to save data
_SAVE = os.path.join("Results", "dynamics")

if not os.path.exists(_SAVE):

    os.makedirs(_SAVE)

##############################################################################
# SIMULATION PARAMTERS
##############################################################################

dt = args.dt
fsamp = 1 / dt
time = np.arange(-4, 2, 1 / fsamp)
beta = 1e-4
Npoints = len(time)
decim = 15
ntrials = args.ntrials
f = args.f  # frequency of oscillation
a = args.a  # hopf parameter, controlling how far each node is from the bifurcation
g = args.g  # global coupling
eta = args.eta


##############################################################################
# CREATE NETWORK ADJ. MATRIX
##############################################################################

data = np.load("interareal/markov2014.npy", allow_pickle=True).item()
A = data["FLN"].T
# Hierarchy values
h = np.squeeze(data["Hierarchy"].T)
A = (1 + eta * h)[:, np.newaxis] * A
Nareas = len(A)

area_names = range(Nareas)


##############################################################################
# CREATE EXTERNAL INPUT
##############################################################################

# Create External Input matrix for model to node 1
input_ext_n1 = (time > 0) * (time < 0.4)
Iext = np.zeros((ntrials, Nareas, Npoints))  # external stimulus
Iext[:, 0, :] = input_ext_n1  # instead of 0, put n

# Create CS matrix

Amplitudes = np.linspace(0, 0.05, ntrials)
# this is the scaling amplitude over trials
CS = Amplitudes[:, None, None] * Iext


##############################################################################
# SIMULATE
##############################################################################


def main(rep):

    seeds = np.random.randint(0, 10000, ntrials)
    data = []

    for trial in tqdm(range(ntrials)):
        data += [
            np.real(
                simulate(
                    A,
                    g,  # global coupling
                    f,
                    a,  # hopf parameter, controlling how far each node is from the bifurcation
                    fsamp,
                    beta,
                    Npoints,
                    CS[trial],  # here goes the stimulus amplitude
                    seeds[trial],
                    "cpu",
                    decim,
                    "both",
                )
            )
        ]

    data = xr.DataArray(
        data,
        dims=("trials", "roi", "times"),
        coords=(range(ntrials), area_names, time[::decim]),
    ).sel(times=slice(-1, 2))

    jax.clear_caches()

    return data


data = main(0)


##############################################################################
# save data to disc
##############################################################################

data.to_netcdf(
    os.path.join(
        _SAVE,
        f"data_f_{f}_a_{a}_g_{g}_eta_{eta}_seed_{seed}.nc",
    )
)
