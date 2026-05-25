"""SIR epidemic-model calibration as a real-world PSO use case.

The objective is to recover the three parameters of a Susceptible-Infected-
Recovered (SIR) model — contact rate beta, recovery rate gamma and initial
infected count I0 — from a noisy time series of daily infections. This is a
classic *inverse problem* in epidemiology: the model is known, the
observations are given, and we look for the parameters that minimise the
mean squared error between simulated and observed curves.

Why this is a good use case for the parallelism study:

- Each evaluation integrates a small ODE for ~100 days. Wall cost is in the
  10-50 ms range, which is firmly above the IPC overhead threshold. V2
  multiprocessing should finally win, in contrast to the 1-microsecond
  benchmark fitnesses where V0 dominated.
- The simulation can terminate early when the disease dies out (no
  infected left). That gives V3 (asyncio) a *natural* asymmetric workload
  without needing to inject artificial latency.
- The model is pure NumPy arithmetic, so a vectorised variant evaluating
  all particles in parallel is straightforward — V4 still wins.
"""
from pathlib import Path
from typing import Optional

import numpy as np


# Fixed simulation grid. Kept module-level so both the scalar and the
# vectorised variant share exactly the same time discretisation.
#
# _DT is deliberately small (sub-day step). Two reasons:
#   1. Numerical accuracy — explicit Euler with dt=1.0 has noticeable
#      drift relative to a finer integration; dt=0.005 gives a curve
#      almost indistinguishable from RK4.
#   2. Wall-clock cost — each evaluation now runs ~20 000 inner steps,
#      lifting the per-particle fitness cost into the ~5–10 ms range
#      where multiprocessing (V2) finally has a chance against V0.
_N_POPULATION = 1_000_000
_N_DAYS = 100
_DT = 0.005
_STEPS_PER_DAY = int(round(1.0 / _DT))

# Physical ranges for each parameter. The PSO works on a normalised
# unit cube [0, 1]^3 and the fitness rescales each coordinate before
# integrating. This keeps PSOConfig's single (lower, upper) pair valid
# across all three dimensions and lets the swarm explore a uniform
# search box even though the underlying physics has heterogeneous
# ranges.
_RANGES = np.array([
    [0.0, 1.0],     # beta  (transmission rate, typical 0..1)
    [0.0, 1.0],     # gamma (recovery rate, typical 0..1)
    [1.0, 100.0],   # I0    (initial infected, must be >= 1)
])


def _denormalise(theta_unit: np.ndarray) -> tuple[float, float, float]:
    """Map a [0, 1]^3 vector back to (beta, gamma, I0) in physical units."""
    lo = _RANGES[:, 0]
    hi = _RANGES[:, 1]
    phys = lo + theta_unit * (hi - lo)
    return float(phys[0]), float(phys[1]), float(phys[2])


# Lazy-loaded observations: read once on first call from CSV. Avoids
# making every fitness evaluation hit the disk.
_OBSERVATIONS: Optional[np.ndarray] = None
_OBS_PATH = Path(__file__).resolve().parents[3] / "data" / "sir_observations.csv"


def _load_observations() -> np.ndarray:
    global _OBSERVATIONS
    if _OBSERVATIONS is None:
        _OBSERVATIONS = np.loadtxt(_OBS_PATH, delimiter=",", skiprows=1, usecols=1)
    return _OBSERVATIONS


def simulate_sir(beta: float, gamma: float, I0: float,
                 n_days: int = _N_DAYS, N: int = _N_POPULATION,
                 dt: float = _DT) -> np.ndarray:
    """Integrate one SIR trajectory and return the daily infected curve.

    Uses an explicit Euler step with sub-day resolution. The returned
    array has shape (n_days+1,) with I(t) sampled at integer days
    t = 0, 1, ..., n_days. Integration stops early once the disease is
    extinct — this is what makes the workload naturally asymmetric for
    V3 (a parameter that produces a doomed epidemic finishes in a few
    milliseconds; a runaway epidemic uses the full budget).
    """
    steps_per_day = int(round(1.0 / dt))
    S = float(N - I0)
    I = float(I0)
    R = 0.0
    curve = np.zeros(n_days + 1)
    curve[0] = I
    for day in range(n_days):
        if I < 1.0:                        # disease extinct, no further dynamics
            break
        for _ in range(steps_per_day):
            new_inf = beta * S * I / N
            new_rec = gamma * I
            S -= new_inf * dt
            I += (new_inf - new_rec) * dt
            R += new_rec * dt
            if I < 0.0:
                I = 0.0
        curve[day + 1] = I
    return curve


def sir(theta: np.ndarray) -> float:
    """Scalar fitness for one normalised parameter vector theta in [0,1]^3.

    The three coordinates are rescaled internally to physical units via
    `_denormalise`. Returns the mean squared error between the simulated
    daily-infected curve and the observed one (normalised by total
    population so the magnitude stays sensible regardless of N).
    """
    theta = np.asarray(theta, dtype=float)
    beta, gamma, I0 = _denormalise(theta)
    obs = _load_observations()
    sim = simulate_sir(beta, gamma, I0)
    diff = (sim - obs) / _N_POPULATION
    return float(np.mean(diff ** 2))


def sir_vec(Theta: np.ndarray) -> np.ndarray:
    """Vectorised fitness — integrates all particles in parallel.

    Theta has shape (n_particles, 3) and is expected in [0, 1]^3. The
    Euler loop is over time, not particles, so the inner step is a
    vector-vector NumPy op of size n_particles. No early termination
    here (the cost of branching across particles would lose more than it
    saves), so V4's run time is deterministic in the number of days.
    """
    Theta = np.asarray(Theta, dtype=float)
    lo = _RANGES[:, 0]
    hi = _RANGES[:, 1]
    phys = lo + Theta * (hi - lo)          # broadcast (n, 3) * (3,) -> (n, 3)
    beta = phys[:, 0]
    gamma = phys[:, 1]
    I0 = phys[:, 2]

    n = Theta.shape[0]
    S = np.full(n, float(_N_POPULATION)) - I0
    I = I0.copy()
    curves = np.zeros((n, _N_DAYS + 1))
    curves[:, 0] = I

    for day in range(_N_DAYS):
        for _ in range(_STEPS_PER_DAY):
            new_inf = beta * S * I / _N_POPULATION
            new_rec = gamma * I
            S -= new_inf * _DT
            I = np.maximum(I + (new_inf - new_rec) * _DT, 0.0)
        curves[:, day + 1] = I

    obs = _load_observations()                             # shape (n_days+1,)
    diff = (curves - obs[np.newaxis, :]) / _N_POPULATION
    return np.mean(diff ** 2, axis=-1)
