[![PyPI](https://img.shields.io/pypi/v/scVAEIT?label=pypi&color=orange)](https://pypi.org/project/scVAEIT)
[![PyPI-Downloads](https://img.shields.io/pepy/dt/scVAEIT?color=green)](https://pepy.tech/project/scVAEIT)

# Variational autoencoder for multimodal mosaic integration and transfer learning

This repository contains implementations of *scVAEIT* for integration and imputation of multi-modal datasets. 
*scVAEIT* (Variational autoencoder for multimodal single-cell mosaic integration and transfer learning) was originally proposed by [[Du22]](#references) for single-cell genomics data.
*scVAEIT* is a deep generative model based on a variational autoencoder (VAE) with masking strategies, which can integrate and impute multi-modal single-cell data, such as single-cell DOGMA-seq, CITE-seq, and ASAP-seq data. 
*scVAEIT* has also been extended to impute single-cell proteomic data in [[Moon24]](#references), though it is also applicable to other types of data.
*scVAEIT* is implemented in Python, and an R wrapper is also available.

For `R` users, `reticulate` can be used to call `scVAEIT` from `R`.
The documentation and tutorials using both `Python` and `R` are available at [scvaeit.readthedocs.io](https://scvaeit.readthedocs.io/en/latest/).

Check out the example folder for illustrations of how to use *scVAEIT*:

Example | Language | Notebooks
---|---|---
Imputation of ADT | ![Python Badge](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff&style=flat-square) | [`imputation_1modality.ipynb`](https://github.com/jaydu1/scVAEIT/blob/main/docs/tutorial/python/imputation_1modality.ipynb)
Imputation of RNA and ADT | ![Python Badge](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff&style=flat-square) | [`imputation_2modalities.ipynb`](https://github.com/jaydu1/scVAEIT/blob/main/docs/tutorial/python/imputation_2modalities.ipynb)
Integration of RNA, ADT, and peaks | ![Python Badge](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff&style=flat-square) | [`integration_3modalities.ipynb`](https://github.com/jaydu1/scVAEIT/blob/main/docs/tutorial/python/integration_3modalities.ipynb)
Imputation of RNA | ![R Badge](https://img.shields.io/badge/R-276DC3?logo=r&logoColor=fff&style=flat-square) | [`imputation_scRNAseq.ipynb`](https://github.com/jaydu1/scVAEIT/blob/main/docs/tutorial/R/imputation_scRNAseq.ipynb)
Imputation of peptides | ![R Badge](https://img.shields.io/badge/R-276DC3?logo=r&logoColor=fff&style=flat-square) | [`imputation_peptide.ipynb`](https://github.com/jaydu1/scVAEIT/blob/main/docs/tutorial/R/imputation_peptide.ipynb)



For preparing your own data to run scVAEIT, please read about:

Example | Language | Notebooks
---|---|---
Prepare input data | ![Python Badge](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff&style=flat-square) | [`prepare_data_input.ipynb`](https://github.com/jaydu1/scVAEIT/blob/main/docs/tutorial/python/prepare_data_input.ipynb)


## Reproducibility Materials
The code for reproducing results in the paper [[Du22]](#references) can be found in the folder `Reproducibility materials`.
The large preprocessed dataset that contains DOGMA-seq, CITE-seq, and ASAP-seq data from GSE156478 can be accessed through [Google Drive](https://drive.google.com/drive/folders/19bzIGKex9Cwoy3ZWXra6D2hvqDtZOvfB?usp=drive_link).




## Dependencies

The package can be installed via PyPI:

```cmd
pip install scVAEIT
```

Alternatively, the dependencies can be installed via the following commands:

```cmd
mamba create --name tf python=3.9 -y
conda activate tf
mamba install -c conda-forge "tensorflow>=2.12, <2.16" "tensorflow-probability>=0.12, <0.24" pandas jupyter -y
mamba install -c conda-forge "scanpy>=1.9.2" matplotlib scikit-learn -y
```

If you are using `conda`, simply replace `mamba` above with `conda`.

The code is only tested on Linux and MacOS. If you are using Windows, installing the dependencies `pip` instead of `conda` is more convenient.



# References


- [Du22] Du, J. H., Cai, Z., & Roeder, K. (2022). Robust probabilistic modeling for single-cell multimodal mosaic integration and imputation via scVAEIT. Proceedings of the National Academy of Sciences, 119(49), e2214414119.
- [Moon24] Moon, H., Du, J. H., Lei, J., & Roeder, K. (2024). Augmented Doubly Robust Post-Imputation Inference for Proteomic data. bioRxiv, 2024-03.
