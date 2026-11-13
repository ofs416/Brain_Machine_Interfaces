# Brain_Machine_Interfaces

Coursework for Cambridge 4G10 (Brain-Machine Interfaces).

## Coursework 1: Neural population dynamics
`Coursework_1/code.ipynb` analyses motor cortex PSTHs (`psths.npz`). It plots raw PSTHs, normalises and mean-centres them, reduces them with PCA, and fits a linear dynamical model (antisymmetric `A`) by maximum likelihood. Figures are in `Figures/` and the report is `4G10_Coursework_1_5553C.pdf`.

## Coursework 2: Decoding hand velocity
Decodes hand velocity from neural activity. Each script downloads the dataset from the course server:
- `Simple.py`: Gaussian smoothing, then a ridge (MAP) linear decoder
- `Wiener.py`: Wiener filter with lagged inputs
- `Kalman.py`: Kalman filter with an autoregressive kinematic prior

Predictions are saved to `Results/` and the report is `4G10_Coursework_2_5553C.pdf`.

## Setup
```
pip install numpy scipy matplotlib scikit-learn requests seaborn jupyter
```
