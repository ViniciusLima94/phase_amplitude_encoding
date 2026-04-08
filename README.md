# Phase-Amplitude Encoding 

This repository investigates the role of **phase and amplitude encoding of stimuli in neuronal dynamics**, with a focus on oscillator-based models. The project uses numerical simulations and statistical analyses to study how phase and amplitude variables contribute to information encoding and inter-areal communication in neural systems.

---

## Scientific Motivation

Neuronal activity is inherently oscillatory, and neural signals are commonly characterized by their **phase** and **amplitude**. While phase-only models capture synchronization and timing relationships, they neglect amplitude fluctuations that play a central role in stimulus encoding and cross-frequency interactions. In particular, **phase–amplitude relationships** have been widely observed in electrophysiological recordings and are thought to support neural communication and information transfer.

This project explores these ideas using **Hopf oscillator dynamics**, which naturally incorporate both amplitude and phase degrees of freedom. By simulating coupled systems and computing relevant statistics, the repository aims to clarify how phase and amplitude jointly encode information in neural signals.

---

## Repository Structure

phase_amplitude_encoding/
├── Figures/
├── interareal/
├── notebooks/
├── src/
├── compute_statistics.py
├── generate_hopf_dynamics.py
├── two_nodes.py
├── run.sh
├── README.md


---

## Core Scripts

### `generate_hopf_dynamics.py`

Generates simulated neural time series using **Hopf oscillator models**.  
This script implements the numerical integration of oscillator dynamics and produces signals with explicit phase and amplitude components. It serves as the primary data-generation step in the analysis pipeline.

---

### `run.sh`

Shell script for automating the workflow, typically executing:
1. Dynamical simulations
2. Statistical analysis
3. Logging of results


---

## Source Code (`src/`)

The `src/` directory contains reusable modules and helper functions, including:
- Numerical integration routines
- Oscillator and coupling definitions
- Utility functions for data handling and analysis

These components are shared across scripts and notebooks to ensure consistency and modularity.

---

## Notebooks (`notebooks/`)

Jupyter notebooks provide exploratory and interpretative analyses, including:
- Visualization of phase and amplitude dynamics
- Parameter sweeps and sensitivity analyses
- Statistical summaries of encoding metrics

---

## Figures (`Figures/`)

Contains figures generated from simulations and analyses.

---

## Interareal Analyses (`interareal/`)

Structural connectome data from Markov et. al., 2013.

---

## Figure-Generating Notebooks

The notebooks whose names start with `FigureXX` are used to generate the **corresponding figures in the associated manuscript**. Each notebook reproduces one main figure by running the relevant simulations, computing summary statistics, and producing publication-quality plots.

The naming convention follows the paper figure numbering (e.g., `Figure01.ipynb` generates Figure 1).

These notebooks are not exploratory; they are **deterministic figure pipelines** intended for reproducibility.

---

### `Figure01.ipynb` 


---

### `Figure02.ipynb` 

---

### `Figure03.ipynb`

# Phase–Amplitude Encoding in Neural Dynamics

> **Reproducibility guide** — how to re-generate every figure from the manuscript.

This repository contains the code, simulations, and analyses accompanying the paper on the role of **phase and amplitude encoding of stimuli in neuronal dynamics**. The study uses coupled Hopf oscillator models to investigate how phase and amplitude jointly encode information and support inter-areal communication in neural systems.

---

## Repository layout

```
phase_amplitude_encoding/
├── notebooks/              # Figure notebooks (FigureXX.ipynb) + exploratory analyses
├── src/                    # Shared Python modules (oscillator definitions, utilities)
├── interareal/             # Structural connectome data (Markov et al., 2013)
├── Results/                # Pre-computed outputs used by the notebooks
├── Figures/                # Output directory — PDFs written here by the notebooks
├── generate_hopf_dynamics.py   # Step 1: simulate Hopf oscillator time series
├── run.sh                      # Convenience wrapper that runs the full pipeline
└── area_coordinates.txt        # Cortical area coordinates for network plots
```

---

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/ViniciusLima94/phase_amplitude_encoding.git
cd phase_amplitude_encoding
```

### 2. Set up the environment

A standard scientific Python stack is required. Using conda:

```bash
conda create -n phase_amp python=3.10
conda activate phase_amp
pip install numpy scipy matplotlib jupyterlab xarray frites
```

> If a `requirements.txt` or `environment.yml` is present at the root of the repo, prefer:
> ```bash
> pip install -r requirements.txt
> # or
> conda env create -f environment.yml
> ```

### 3. Generate the simulated data

The notebooks load pre-computed Hopf oscillator time series from the `Results/` directory. To regenerate them from scratch, run:

```bash
python generate_hopf_dynamics.py
```

Or use the shell script to run the full simulation + analysis pipeline in one step:

```bash
bash run.sh
```

> **Note:** `Results/` may already contain the outputs needed to run the notebooks directly. If so, you can skip this step and open the notebooks immediately.

### 4. Launch Jupyter

```bash
jupyter lab
# or
jupyter notebook
```

Navigate to the `notebooks/` directory.

---

## Reproducing the figures

Each `FigureXX.ipynb` notebook is a **self-contained, deterministic pipeline** that reads from `Results/`, runs the relevant statistical analyses, and writes a publication-quality PDF to `Figures/`. Run the cells top-to-bottom; no extra setup is needed beyond the environment above.

| Notebook | Output file | What it shows |
|---|---|---|
| `Figure01.ipynb` | `Figures/Figure1.pdf` | Hopf oscillator dynamics — single-node phase & amplitude time series |
| `Figure02.ipynb` | `Figures/Figure2.pdf` | Two-node coupling: phase and amplitude encoding of a stimulus |
| `Figure04.ipynb` | `Figures/Figure4.pdf` | Inter-areal network + oscillatory traces + redundancy/synergy (2- & 3-plets) |
| `Figure05.ipynb` | `Figures/Figure5.pdf` | Summary statistics across the full connectome |

> Figure numbers above match the manuscript. Open the notebook to confirm the exact output filename — some notebooks write both a `.pdf` and a `.png`.

### Running a single notebook non-interactively

```bash
jupyter nbconvert --to notebook --execute notebooks/Figure04.ipynb \
    --output notebooks/Figure04_executed.ipynb
```

### Running all figure notebooks in sequence

```bash
for nb in notebooks/Figure*.ipynb; do
    echo "Running $nb ..."
    jupyter nbconvert --to notebook --execute "$nb" --inplace
done
```

---

## Key source modules (`src/`)

| Module | Purpose |
|---|---|
| Oscillator / coupling definitions | Implements the Hopf normal-form equations and inter-node coupling |
| Numerical integration | Fixed-step or adaptive ODE solvers used by `generate_hopf_dynamics.py` |
| Information-theoretic measures | Mutual information, redundancy, and synergy estimators (2- and 3-plets) |
| Plotting utilities | `plot.Background`, `plot.add_panel_letters`, and other helpers shared across figure notebooks |

---

## Data

**Simulated data** is produced by `generate_hopf_dynamics.py` and stored under `Results/`.

**Structural connectivity** (FLN matrix and area coordinates) is derived from the macaque cortical hierarchy reported in Markov et al. (2013) and is included in the `interareal/` directory.

No proprietary or access-restricted datasets are required to reproduce the figures.
