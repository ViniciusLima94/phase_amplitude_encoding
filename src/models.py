import jax
import jax.numpy as jnp
import numpy as np
from .models_setup import _set_nodes, _set_nodes_delayed
from .utils import randn
from functools import partial

jax.config.update("jax_enable_x64", True)


def _check_params(Iext: jnp.ndarray, N: int):
    if isinstance(Iext, (int, float)):
        return jnp.ones((1, N)) * Iext  # Assure it is a jax ndarray
    elif Iext is None:
        return jnp.zeros((1, N))
    return jnp.asarray(Iext)


# @partial(jax.vmap, in_axes=(0, 0, 0))
def _ode(Z: np.complex128, a: float, w: float):
    return Z * (a + 1j * w - jnp.abs(Z) ** 2)


def simulate(
    A: np.ndarray,
    g: float,
    f: float,
    a: float,
    fs: float,
    eta: float,
    T: float,
    Iext: np.ndarray = None,
    seed: int = 0,
    device: str = "cpu",
    decim: int = 1,
    stim_mode: str = "amp",
):
    """
    Simulates a network of coupled oscillators with external stimulation.

    Parameters:
    ----------
    A : np.ndarray
        Adjacency matrix representing network connectivity.
    g : float
        Coupling strength parameter.
    f : float
        Natural frequency of oscillators.
    a : float
        Nonlinear parameter influencing oscillator dynamics.
    fs : float
        Sampling frequency.
    eta : float
        Noise intensity.
    T : float
        Total simulation time in discrete steps.
    Iext : np.ndarray, optional
        External input to the oscillators (default is None, meaning no input).
    seed : int, optional
        Random seed for noise generation (default is 0).
    device : str, optional
        Computational device, either "cpu" or "gpu" (default is "cpu").
    decim : int, optional
        Decimation factor for downsampling the output (default is 1, meaning no downsampling).
    stim_mode : str, optional
        Stimulation mode, can be "amp" (amplitude), "phase", or "both" (default is "amp").

    Returns:
    -------
    np.ndarray
        Array of oscillator phases over time, with shape (N, T/decim), where N is the number of nodes.

    Notes:
    ------
    - Uses JAX for optimized computation, supporting CPU and GPU execution.
    - Implements stochastic differential equations for phase oscillator dynamics.
    """

    assert stim_mode in ["amp", "phase", "both"]
    assert device in ["cpu", "gpu"]

    jax.config.update("jax_platform_name", device)

    N, A, omegas, phases_history, dt, a = _set_nodes(A, f, fs, a)

    g = _check_params(g, T).squeeze()
    eta = _check_params(eta, N).squeeze()
    Iext = _check_params(Iext, N)

    # Stim parameters
    gain = 0
    phi = 0
    offset = 1

    if stim_mode == "amp":
        gain = 1
        offset = 0
    elif stim_mode == "phase":
        gain = 1
        phi = np.pi / 2
        offset = 0

    times = np.arange(T, dtype=int)  # Time array

    # Scale with dt to avoid doing it evert time-step
    A = A * dt
    eta = eta * jnp.sqrt(dt)
    Iext = Iext * dt

    # @jax.jit
    def _loop(carry, t):

        phases_history = carry

        phases_t = phases_history.squeeze().copy()

        phase_differences = phases_t - phases_history

        exp_phi = gain * jnp.exp(1j * (jnp.angle(phases_t) + phi)) + offset

        # Input to each node
        Input = g[t] * (A * phase_differences).sum(axis=1) + Iext[:, t] * exp_phi

        phases_history = phases_history.at[:, 0].set(
            phases_t
            + dt * _ode(phases_t, a, omegas)
            + Input
            + eta * randn(size=(N,), seed=seed + t)
            + eta * 1j * randn(size=(N,), seed=seed + t + 2 * t)
        )

        carry = jax.lax.reshape(phases_history, (N, 1))
        return carry, phases_history

    _, phases = jax.lax.scan(_loop, (phases_history), times)

    return phases[::decim].squeeze().T


def simulate_kuramoto(
    A: np.ndarray,
    D: np.ndarray,
    g: float,
    f: float,
    fs: float,
    eta: float,
    T: float,
    Iext: np.ndarray = None,
    seed: int = 0,
    device: str = "cpu",
    decim: int = 1,
):
    """
    Simulates a network of coupled oscillators with external stimulation.

    Parameters:
    ----------
    A : np.ndarray
        Adjacency matrix representing network connectivity.
    D : np.ndarray
        Delay matrix in time-steps
    g : float
        Coupling strength parameter.
    f : float
        Natural frequency of oscillators.
    fs : float
        Sampling frequency.
    eta : float
        Noise intensity.
    T : float
        Total simulation time in discrete steps.
    Iext : np.ndarray, optional
        External input to the oscillators (default is None, meaning no input).
    seed : int, optional
        Random seed for noise generation (default is 0).
    device : str, optional
        Computational device, either "cpu" or "gpu" (default is "cpu").
    decim : int, optional
        Decimation factor for downsampling the output (default is 1, meaning no downsampling).

    Returns:
    -------
    np.ndarray
        Array of oscillator phases over time, with shape (N, T/decim), where N is the number of nodes.

    Notes:
    ------
    - Uses JAX for optimized computation, supporting CPU and GPU execution.
    - Implements stochastic differential equations for phase oscillator dynamics.
    """

    assert device in ["cpu", "gpu"]

    jax.config.update("jax_platform_name", device)

    N, A, D, max_delay, omegas, _, dt, _ = _set_nodes_delayed(A, f, fs, 0)

    # Normalize by the number of nodes (see Kuramoto equation)
    g = _check_params(g, T).squeeze() / N
    eta = _check_params(eta, N).squeeze()
    Iext = _check_params(Iext, N)

    times = np.arange(T, dtype=int)  # Time array

    # Scale with dt to avoid doing it evert time-step
    A = A * dt
    eta = eta * jnp.sqrt(dt)
    Iext = Iext * dt
    omegas *= dt

    # For Kuramoto phases_history is cast to float
    # Randomly initialize phases and keeps it only up to max delay
    phases_history = (
        2 * np.pi * np.random.rand(N, max_delay) + omegas[:, None] * np.arange(1, 2)
    ) % (2 * np.pi)

    # Nodes indexes
    nodes = jnp.arange(N)

    # @jax.jit
    def _loop(carry, t):

        phases_history = carry

        # phases_t = phases_history.squeeze().copy()
        # phase_differences = jnp.sin(phases_t - phases_history)

        phases_t = phases_history[:, -1].copy()

        @partial(jax.vmap, in_axes=(0, 0))
        def _return_phase_differences(n, d):
            return jnp.sin(phases_history[np.indices(d.shape)[0], d - 1] - phases_t[n])

        phase_differences = _return_phase_differences(nodes, D)

        Input = g[t] * (A * phase_differences).sum(axis=1) + Iext[:, t]

        phases_history = phases_history.at[:, :-1].set(phases_history[:, 1:])

        phases_history = phases_history.at[:, -1].set(
            phases_t + omegas + Input + eta * randn(size=(N,), seed=seed + t)
        )

        # carry = jax.lax.reshape(phases_history, (N, 1))
        carry = phases_history
        return carry, phases_history[:, -1]

    _, phases = jax.lax.scan(_loop, (phases_history), times)

    phases_fft = jnp.fft.fft(jnp.sin(phases), n=T, axis=0)
    phases = jnp.fft.ifft(phases_fft, axis=0).real

    return phases[::decim].squeeze().T


def simulate_delayed(
    A: np.ndarray,
    D: np.ndarray,
    g: float,
    f: float,
    a: float,
    fs: float,
    eta: float,
    T: float,
    Iext: np.ndarray = None,
    seed: int = 0,
    device: str = "cpu",
    decim: int = 1,
    stim_mode: str = "amp",
):

    assert stim_mode in ["amp", "phase", "both"]
    assert device in ["cpu", "gpu"]

    jax.config.update("jax_platform_name", device)

    N, A, D, omegas, phases_history, dt, a = _set_nodes_delayed(A, D, f, fs, a)

    g = _check_params(g, T).squeeze()
    eta = _check_params(eta, N).squeeze()
    Iext = _check_params(Iext, N)

    # Stim parameters
    gain = 0
    phi = 0
    offset = 1

    if stim_mode == "amp":
        gain = 1
        offset = 0
    elif stim_mode == "phase":
        gain = 1
        phi = np.pi / 2
        offset = 0

    times = np.arange(T, dtype=int)  # Time array

    # Scale with dt to avoid doing it evert time-step
    A = A * dt
    eta = eta * jnp.sqrt(dt)
    Iext = Iext * dt

    nodes = jnp.arange(N)

    @jax.jit
    def _loop_delayed(carry, t):

        phases_history = carry

        phases_t = phases_history[:, -1].copy()

        @partial(jax.vmap, in_axes=(0, 0))
        def _return_phase_differences(n, d):
            return phases_history[np.indices(d.shape)[0], d - 1] - phases_t[n]

        phase_differences = _return_phase_differences(nodes, D)

        # phase_differences = np.stack(
        #    [_return_phase_differences(n, d) for n, d in enumerate(D)]
        # )

        exp_phi = gain * jnp.exp(1j * (jnp.angle(phases_t) + phi)) + offset

        # Input to each node
        Input = (g[t] * A * phase_differences).sum(axis=1) + Iext[:, t] * exp_phi

        phases_history = phases_history.at[:, :-1].set(phases_history[:, 1:])

        phases_history = phases_history.at[:, -1].set(
            phases_t
            + dt * _ode(phases_t, a, omegas)
            + Input
            + eta * randn(size=(N,), seed=seed + t)
            + eta * 1j * randn(size=(N,), seed=seed + t + 2 * t)
        )

        carry = phases_history  # jax.lax.reshape(phases_history, (N, max_delay))
        return carry, phases_history[:, -1]

    _, phases = jax.lax.scan(_loop_delayed, (phases_history), times)

    return phases[::decim].squeeze().T
