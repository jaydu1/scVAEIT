# Variational autoencoder for multimodal single-cell mosaic integration and transfer learning.

This repository contains implementations of *scVAEIT* for the paper ''*Robust probabilistic modeling for single-cell multimodal mosaic integration and imputation via scVAEIT*'' ([bioRxiv](https://doi.org/10.1101/2022.07.25.501456)).


Check out the example folder for illustrations of how to use *scVAEIT* for
- imputation [`integration_2modalities.ipynb`](https://github.com/jaydu1/scVAEIT/blob/main/example/imputation_2modalities.ipynb)
- integration [`integration_3modalities.ipynb`](https://github.com/jaydu1/scVAEIT/blob/main/example/integration_3modalities.ipynb)

# Dependencies

The dependencies can be installed via the following commands:

```cmd
mamba create --name tf python=3.9 -y
conda activate tf
mamba install -c conda-forge "tensorflow>=2.12" "tensorflow-probability>=0.12" pandas jupyter -y
mamba install -c conda-forge "scanpy>=1.9.2" matplotlib scikit-learn -y
```

If you are using `conda`, simply replace `mamba` above by `conda`.


# Reproducibility Materials
The code for reproducing results in the paper can be found at the folder `Reproducibility materials`.

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
