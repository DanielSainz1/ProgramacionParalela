"""Generate the synthetic SIR ground-truth observations used by the use case.

Run this once to produce data/sir_observations.csv. The CSV stores a daily
time series of new-infected counts produced by simulating the SIR model
with known ground-truth parameters and adding 5% multiplicative Gaussian
noise. The PSO fitness then recovers those parameters from the noisy
curve — this is the standard 'twin experiment' validation used in
epidemiology before applying methods to real data.

Ground truth: beta=0.30, gamma=0.10, I0=10. Hidden from the PSO, of course.
"""
from pathlib import Path

import numpy as np

from pso.objectives.sir import simulate_sir, _N_DAYS


GROUND_TRUTH = dict(beta=0.30, gamma=0.10, I0=10.0)
NOISE_RELATIVE_STD = 0.05
SEED = 20260525            # date-of-generation seed, locked in for reproducibility


def main():
    rng = np.random.default_rng(SEED)
    clean = simulate_sir(GROUND_TRUTH["beta"], GROUND_TRUTH["gamma"], GROUND_TRUTH["I0"])
    noise = rng.normal(loc=1.0, scale=NOISE_RELATIVE_STD, size=clean.shape)
    noisy = np.maximum(clean * noise, 0.0)        # multiplicative noise, clipped to >=0

    out = Path("data/sir_observations.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write("day,infected\n")
        for t, y in enumerate(noisy):
            f.write(f"{t},{y:.4f}\n")
    print(f"Wrote {out} ({len(noisy)} rows) with ground truth {GROUND_TRUTH}")


if __name__ == "__main__":
    main()
