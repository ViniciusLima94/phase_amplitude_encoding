# phase_amplitude_encoding

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

---

### `Figure04.ipynb` 


---

### `Figure05.ipynb` 



---

## Reproducibility Notes

- Each `FigureXX` notebook is self-contained and can be run independently.
- Parameters are fixed to match those used in the manuscript.
- Figures produced by these notebooks correspond directly to the paper figures, aside from minor stylistic differences (e.g., font size).



## Running the Code

1. Clone the repository:
   ```bash
   git clone https://github.com/ViniciusLima94/phase_amplitude_encoding.git
   cd phase_amplitude_encoding


Install dependencies (typical scientific Python stack):

 - pip install numpy scipy matplotlib jupyter


