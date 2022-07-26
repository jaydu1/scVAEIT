# Variational autoencoder for multimodal single-cell mosaic integration and transfer learning.

This repository contains data and codes to reproduce results in the paper ''*Robust probabilistic modeling for single-cell multimodal mosaic integration and imputation via scVAEIT*'' ([bioRxiv](https://doi.org/10.1101/2022.07.25.501456)).


# Requirement


Python packages for running `scVAEIT`

```
python                    3.8.12
scanpy                    1.7.2
scikit-learn              1.0.2
tensorflow                2.4.1
tensorflow-addons         0.13.0
tensorflow-gpu            2.4.1
tensorflow-probability    0.13.0 
```


Python packages for running `totalVI` and `MultiVI`

```
python                    3.9.12
pytorch                   1.10.2
pytorch-gpu               1.10.2
pytorch-lightning         1.5.10
scanpy                    1.8.2
scvi-tools                0.15.0
```


Python packages for running experiments and plotting:

```
h5py                      2.10.0
hdf5                      1.10.5
matplotlib                3.5.1
numpy                     1.21.5
pandas                    1.4.1
python                    3.8.12
seaborn                   0.11.2
scipy                     1.8.0
```

R packages for running `Seurat` (installed through anaconda)

```
r-dplyr                   1.0.7
r-harmony                 0.1
r-hdf5r                   1.3.5
r-reticulate              1.24
r-seurat                  4.1.0
r-seuratdisk              0.0.9019
r-seuratobject            4.0.4
r-shiny                   1.7.1
r-signac                  1.2.1
```


# Files

- `./data/` contains raw and preprocessed data, as well as the instruction file.
- `./ex1_bimodal/` contains scripts for bimodal experiments of a CITE-seq PBMC dataset, a CITE-seq CBMC dataset, and a REAP-seq PBMC dataset.

	- `pbmc_Mono_dimlatent.py`: Experiments with varying latent dimensions on the Mono cell type of the CITE-seq dataset from Seurat v4's paper (Tab. S2).

	- `pbmc_scVAEIT.py`, `pbmc_totalVI.py`, and `pbmc_Seurat.R`: They require an integer (0-1 for Mono and CD4 T) as input to the program (Fig. 1a).

	- `pbmc_external_scVAEIT.py`, `pbmc_external_totalVI.py`, and `pbmc_external_Seurat.R`: Experiments on external datasets (Fig. 1b-c).

- `./ex2_trimodal/` contains scripts for trimodal experiments of a DOGMA-seq PBMC dataset.

	- `dogma_scVAEIT.py`, `dogma_MultiVI.py`, `dogma_totalVI.py`, and `dogma_Seurat.R`: Experiments on trimodal datasets (Fig. 3 and Fig. 4).

- `./ex3_intermediate_integration/` contains scripts for trimodal intermediate integration.

	- `dogma_int_scVAEIT.py` and `dogma_int_Seurat.R`: Experiments on two-phase mosaic integration (Fig. 5a).

	- `dogma_int_scVAEIT_full.py`: Experiment of intermediate integration of a DOGMA-seq PBMC dataset, a CITE-seq PBMC dataset and an ASAP-seq PBMC dataset (Fig. 5b).

- `plot.py` produces the figures.