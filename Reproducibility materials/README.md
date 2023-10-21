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

- `./ex4_masking_prob/` contains a script for the sensitivity analysis of the masking probability (Fig. S8).

- `./ex5_multi_single_modal/` contains a script for the comparison with and without single-modal datasets included (Fig. S9).

- `plot.py` produces the figures.
  


# Parameters
## Network parameters

In the example, basically, the network is operated in two levels of blocks:
- The feature levels: the number of genes $n_g$, the number of adts $n_a$, the number of peaks $n_p$. The related parameters are `dim_input_arr` and `uni_block_names` (meaning that they have the same length).
- The subconnection level: the number of genes, the number of ADTs, the number of peaks in chrom 1 $n_p^1$, the number of peaks in chrom 2 $n_p^2$, etc. The related parameters include `dist_block`, `dim_block_embed`, `dim_block_enc`, `dim_block_dec`, and `block_names`.

We explain the parameters as below:

- `dim_input_arr` represents the size of input features. In the example, it is simply $[n_g, n_a, n_p]$.

- `dim_block` represents the number of subconnected features in all modalities (assuming that the features have been rearranged accordingly). In the example, it is $[n_g, n_a, n_p^1, n_p^2, \ldots]$. 

- `dist_block`: There are four distributions implemented: 'NB', 'ZINB', 'Bernoulli', 'Gaussian' for negative binomial, zero-inflated negative binomial, Bernoulli, and Gaussian, respectively. However, only 'NB' and 'Bernoulli' are tested and used to generate the results in the paper. 'Bernoulli' is used for ATAC-seq data, and 'NB' is used for genes and proteins.

- `dim_block_embed` represents the embedding dimension of the binary mask. For example, `dim_block_embed = [1, 2, 3, ...]` means the mask will be embedded into a continuous vector of dimension 1 for block 1,  and so on.

- `dim_block_enc` represents the structure of the first latent layer of the encoder. Using skip-connection helps reduce memory and computation complexity. 
In the example, `dim_block_enc = np.array([256, 128] + [16 for _ in chunk_atac])` means that the genes will be embedded into a vector of dimension 256, the adts will be embedded into a vector of dimension 128, and so on. 
For block `i`, we have a sub-network that takes both the features of size `dim_input_arr[i]` and the mask embedding of size `dim_block_embed[i]` and outputs a vector of size `dim_block_enc[i]`.
After that, the embedding vectors in all blocks will be concatenated into a vector as the input to the encoder. 

- Similarly, `dim_block_dec` represents the structure of the last latent layer of the decoder. For block `i`, we have a sub-network that takes latent features of size `dim_block_dec[i]` and outputs a vector (the predicted features) of size `dim_input_arr[i]`.


- `dimensions` and `dim_latent` specify the network structure in the middle. For example, `dimensions = [256, 128]` and `dim_latent = 32` mean that we have a network $n_{in}-256-128-32-128-256-n_{out}$ where $n_{in}$ is the sum of `dim_block_enc`, and $n_{out}$ is the sum of `dim_block_dec`.

## Hyperparameters
Some of the important hyperparameters are:
- `beta_unobs` represents that weight for unobserved features.
- `p_feat` represents the probability of masking for the individual features. The larger value of `p_feat` encourages imputation ability but also requires more training epochs to have a good performance. But the influence of it is not large when training for enough epochs, so we recommend fixing fix `p_feat` as any reasonable value, e.g. 0.2. 
- `p_modal` represents the probability of masking out one modality. You can just leave it as a uniform.

In our experiments, the results were not sensitive to the above parameters. So you can just use reasonable values as in the example, except the following parameter requires some care depending on your data:

- `beta_modal` represents the importance of each modality. You run the model on your dataset for a few epochs and pick `beta_modal` such that the likelihoods (which will be printed during training) of all modalities are roughly in the same order. Notably, the number of peaks is generally very large, so its likelihood will have a higher value. And that is why you can see it has a small weight 0.01, in the example where `beta_modal = [0.14,0.85,0.01]`.


