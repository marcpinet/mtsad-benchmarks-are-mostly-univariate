# Anomalies in Multivariate Time Series Benchmarks Are Mostly Univariate

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running each component

Helper scripts are available in `scripts/`.

## LinearAE and AE

```bash
./scripts/run_ae_cd.sh  # channel-dependent
./scripts/run_ae_ci.sh  # channel-independent
./scripts/run_linearae_cd.sh  # channel-dependent
./scripts/run_linearae_ci.sh  # channel-independent
```

Note that these models also exist under `notebooks/models/` as notebooks, with many different runs.

### CATCH

```bash
./scripts/run_catch.sh
```

Generates a synthetic multivariate signal with injected cross-channel correlation anomalies using our formula, fits the CATCH model and prints AUC-ROC, AUC-PR and the VUS metrics.

### CrossAD

Two variants are available: channel-dependent (`Basic_CrossAD_CD`) and channel-independent (`Basic_CrossAD`):

```bash
./scripts/run_crossad_cd.sh  # channel-dependent
./scripts/run_crossad_ci.sh  # channel-independent
```

Hyperparameters are in `models/crossad/configs/synthetic/`, the script reports the same metrics as CATCH.

## Run everything at once

```bash
./scripts/run_all_synth.sh
```

### Δρ sanity check

```bash
./scripts/run_deltamax_test.sh
```

Checks how many abnormal segments are being detected from our method for synthetic data. For a segment, if at least 1 method (Pearson, Spearman, Distance-Correlation) detects the cross-channel correlation break, it's considered detected.

### Notebooks

```bash
jupyter lab notebooks/
```

- `notebooks/preprocessing/`: one notebook per dataset
- `notebooks/models/`: LinearAE and AE experiments on synthetic data with NPRoll injection

Place datasets under `datasets/<dataset_name_lowercase>/`

- Get GECCO [here](https://zenodo.org/records/3884398) (`datasets/gecco/1_gecco2018_water_quality.csv`)
- Get SWAN and SWaT [here](https://github.com/decisionintelligence/CrossAD) (`datasets/swan/SWAN.csv` and `datasets/swat/SWaT.csv`)

The others are directly retrieved from kaggle.

## Notes

- GPU is auto-detected (CUDA / Apple MPS / CPU) in the runners
- Code was written in Python 3.12
- Notebooks are available in `notebooks/`, already fully run and with the results for you to check.

## Citation

```bibtex
@article{Pinet2026AnomaliesIM,
  title={Anomalies in Multivariate Time Series Benchmarks Are Mostly Univariate},
  author={Marc Pinet and Julien Cumin and Samuel Berlemont and Dominique Vaufreydaz},
  year={2026},
  url={https://doi.org/10.48550/arXiv.2606.02670}
}
```