# Raw Data

The related raw data and corresponding sources are listed below.
We provide most of them in this directory for users' convenience, while a few of them are not compatible with github as they exceed 100Mb and are needed to obtained from the links listed below.

- CITE-seq PBMC
	
	`pbmc_multimodal.h5seurat` from [Seurat v4's tutorial](https://satijalab.org/seurat/articles/multimodal_reference_mapping.html).
	
	Mimitou, E. P., Lareau, C. A., Chen, K. Y., Zorzetto-Fernandes, A. L., Hao, Y., Takeshima, Y., ... & Smibert, P. (2021). Scalable, multimodal profiling of chromatin accessibility, gene expression and protein levels in single cells. Nature biotechnology, 39(10), 1246-1258.


- CITE-seq CBMC

	Dataset `'bmcite'` from `SeuratData`.

	
	Stoeckius, M., Hafemeister, C., Stephenson, W., Houck-Loomis, B., Chattopadhyay, P. K., Swerdlow, H., ... & Smibert, P. (2017). Simultaneous epitope and transcriptome measurement in single cells. Nature methods, 14(9), 865-868.
	
- REAP-seq PBMC

	`REAPseq/GSM2685238_mRNA_2_PBMCs_matrix.txt.gz`

	`REAPseq/GSM2685239_mRNA_3_PBMCs_matrix.txt.gz`

	`REAPseq/GSM2685243_protein_2_PBMCs_matrix.txt.gz`

	`REAPseq/GSM2685244_protein_3_PBMCs_matrix.txt.gz`
	
	Peterson, V. M., Zhang, K. X., Kumar, N., Wong, J., Li, L., Wilson, D. C., ... & Klappenbach, J. A. (2017). Multiplexed quantification of proteins and transcripts in single cells. Nature biotechnology, 35(10), 936-939.

- DOGMA-seq, CITE-seq, and ASAP-seq PBMC

	The following files are retrieved from [Github Repo](https://github.com/caleblareau/asap_reproducibility):

	`pbmc_LLL_processed.rds`

	`stim_fragments.tsv.gz.tbi`

	`stim_fragments.tsv.gz`

	`control_fragments.tsv.gz`

	`control_fragments.tsv.gz.tbi`

	`22July2020_Seurat_Coembed4.rds`

	`ASAP_embedding_CLRadt.rds`


	Mimitou, E. P., Lareau, C. A., Chen, K. Y., Zorzetto-Fernandes, A. L., Hao, Y., Takeshima, Y., ... & Smibert, P. (2021). Scalable, multimodal profiling of chromatin accessibility, gene expression and protein levels in single cells. Nature biotechnology, 39(10), 1246-1258.


# Filtered Data

The two scripts in this directory are used to preprocess the raw data and generate `.h5` files, which are the inputs to Python scripts in the experiments.

- `preprocess_data.R`

	- CITE-seq PBMC 
		
		`pbmc_count.h5`

	- CITE-seq CBMC
		
		`cbmc_count.h5`

	- REAP-seq PBMC

		`reap_count.h5`

	- DOGMA-seq, CITE-seq, and ASAP-seq PBMC
		
		`DOGMA_pbmc.h5`

		`cite_pbmc.h5`

		`asap_pbmc.h5`
	
- `preprocess_data.py` merges `DOGMA_pbmc.h5`, `cite_pbmc.h5`, and `asap_pbmc.h5` to produce `dogma_cite_asap.h5`
	
# Other files


`mask_dogma.csv` contains masks randomly generated to mimic situation when there are only partial measurements.

