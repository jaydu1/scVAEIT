Sys.setenv("OMP_NUM_THREADS" = 4)
Sys.setenv("OPENBLAS_NUM_THREADS" = 4)
Sys.setenv("MKL_NUM_THREADS" = 6)
Sys.setenv("VECLIB_MAXIMUM_THREADS" = 4)
Sys.setenv("NUMEXPR_NUM_THREADS" = 6)

library(Seurat)
library(SeuratDisk)
library(Signac)
library(ggplot2)
library(cowplot)

library(SeuratData)
library(hdf5r)

############################################################
#
# CITE-seq CBMC
#
############################################################
InstallData("bmcite")
bm <- LoadData(ds = "bmcite")
SaveH5Seurat(cbmc, 'bm.h5seurat', overwrite=TRUE)

RNA <- cbmc@assays$RNA@counts
RNA <- RNA[rowSums(RNA)>=500,]
RNA <- RNA[!startsWith(rownames(RNA), 'MOUSE'),]
ADT <- cbmc@assays$ADT@counts
ADT_names <- rownames(ADT)
ADT_names[ADT_names=='CD3'] <- 'CD3-1'
ADT_names[ADT_names=='CD4'] <- 'CD4-2'
ADT_names[ADT_names=='CD56'] <- 'CD56-1'


file.h5 <- H5File$new("cbmc_count.h5", mode = "w")
file.h5[["ADT"]] <- as.matrix(ADT)
file.h5[["RNA.shape"]] <- RNA@Dim
file.h5[["RNA.data"]] <- RNA@x
file.h5[["RNA.indices"]] <- RNA@i
file.h5[["RNA.indptr"]] <- RNA@p
file.h5[["cell_ids"]] <- colnames(RNA)
file.h5[["gene_names"]] <- rownames(RNA)
file.h5[["var_gene_names"]] <- cbmc@assays$RNA@var.features
file.h5[["ADT_names"]] <- ADT_names
file.h5[["celltypes"]] <- as.vector(Idents(cbmc))
file.h5$close_all()



############################################################
#
# REAP-seq PBMC
#
############################################################
reap_rna_2 <- read.table("REAPseq/GSM2685238_mRNA_2_PBMCs_matrix.txt.gz", 
    row.names = 1)
reap_rna_3 <- read.table("REAPseq/GSM2685239_mRNA_3_PBMCs_matrix.txt.gz", 
    row.names = 1)
reap_protein_2 <- read.table("REAPseq/GSM2685243_protein_2_PBMCs_matrix.txt.gz", 
    row.names = 1)
reap_protein_3 <- read.table("REAPseq/GSM2685244_protein_3_PBMCs_matrix.txt.gz", 
    row.names = 1)
RNA <- Matrix::Matrix(as.matrix(cbind(reap_rna_2, reap_rna_3)), sparse = TRUE)
ADT <- Matrix::Matrix(as.matrix(cbind(reap_protein_2, reap_protein_3)), 
    sparse = FALSE)
rownames(ADT) <- as.vector(sapply(rownames(ADT),  function(x)strsplit(x,"_")[[1]][1]))

RNA <- Matrix::Matrix(as.matrix(cbind(reap_rna_2, reap_rna_3)), sparse = TRUE)
ADT <- Matrix::Matrix(as.matrix(cbind(reap_protein_2, reap_protein_3)), 
    sparse = FALSE)
ADT <- ADT[rownames(ADT)!='CD45_TCTCGACT',]

rownames(ADT) <- as.vector(sapply(rownames(ADT),  function(x)strsplit(x,"_")[[1]][1]))
                                  
ADT <- ADT[rownames(ADT)!='Control',]
ADT <- ADT[rownames(ADT)!='Mouse',]
ADT <- ADT[rownames(ADT)!='Blank',]
                                  
adt_assay <- CreateAssayObject(counts = ADT)
reap[["ADT"]] <- adt_assay
reap[["percent.mt"]] <- PercentageFeatureSet(reap, pattern = "^MT-")
reap <- subset(reap, subset = nFeature_RNA >= 200 & percent.mt < 5)

reap <- NormalizeData(reap)
reap <- FindVariableFeatures(reap)
reap <- ScaleData(reap)
reap <- RunPCA(reap, verbose = FALSE)
reap <- FindNeighbors(reap, dims = 1:25)
reap <- FindClusters(reap, resolution = 0.8, verbose = FALSE)
reap <- RunTSNE(reap, dims = 1:25, method = "FIt-SNE")

reap.rna.markers <- FindAllMarkers(reap, max.cells.per.ident = 100, logfc.threshold = log(2), 
    only.pos = TRUE, min.diff.pct = 0.3, do.print = F)

new.cluster.ids <- c('CD4 T', 'CD14+ monocytes', 'CD14+ monocytes', 'CD8 T', 'FCGR3A+ monocytes',
  'B', 'NK', 'B', 'CD14+ monocytes', 'pDCs', 'Megakaryocytes')

names(new.cluster.ids) <- levels(reap)
reap <- RenameIdents(reap, new.cluster.ids)

celltypes <- as.vector(Idents(reap))

ADT_names <- rownames(reap@assays$ADT)
ADT_names[ADT_names=='CD11b'] <- 'CD11b-1'
ADT_names[ADT_names=='CD45'] <- 'CD45-2'
ADT_names[ADT_names=='CD56'] <- 'CD56-1'
ADT_names[ADT_names=='CD158E1'] <- 'CD158e1'
ADT_names[ADT_names=='CD4'] <- 'CD4-2'
ADT_names[ADT_names=='CD4.1'] <- 'CD4-1'
ADT_names[ADT_names=='CD3'] <- 'CD3-2'
ADT_names[ADT_names=='CD8'] <- 'CD8.2'
ADT_names[ADT_names=='CD8.1'] <- 'CD8'

               
RNA <- reap@assays$RNA@counts
ADT <- reap@assays$ADT@counts
rownames(ADT) <- ADT_names
                                  
reap <- CreateSeuratObject(RNA)
adt_assay <- CreateAssayObject(counts = ADT)
reap[["ADT"]] <- adt_assay

file.h5 <- H5File$new("reap_count.h5", mode = "w")
file.h5[["ADT"]] <- as.matrix(ADT)
file.h5[["RNA.shape"]] <- RNA@Dim
file.h5[["RNA.data"]] <- RNA@x
file.h5[["RNA.indices"]] <- RNA@i
file.h5[["RNA.indptr"]] <- RNA@p
file.h5[["cell_ids"]] <- colnames(RNA)
file.h5[["gene_names"]] <- rownames(RNA)
file.h5[["ADT_names"]] <- ADT_names
file.h5[["celltypes"]] <- celltypes
file.h5$close_all()




############################################################
#
# CITE-seq PBMC
#
############################################################
pbmc_multimodal <- LoadH5Seurat("pbmc_multimodal.h5seurat", assays = "counts", reductions = FALSE, graphs = FALSE,)
RNA <- pbmc_multimodal@assays$SCT@counts[VariableFeatures(pbmc_multimodal),]
RNA <- RNA[rowSums(RNA>0)>=500,]
ADT <- pbmc_multimodal@assays$ADT@counts
ADT <- ADT[rowSums(ADT>0)>=500,]

library(hdf5r)
file.h5 <- H5File$new("pbmc_count.h5", mode = "w")
file.h5[["ADT"]] <- as.matrix(ADT)
file.h5[["RNA.shape"]] <- RNA@Dim
file.h5[["RNA.data"]] <- RNA@x
file.h5[["RNA.indices"]] <- RNA@i
file.h5[["RNA.indptr"]] <- RNA@p
file.h5[["cell_ids"]] <- colnames(RNA)
file.h5[["gene_names"]] <- rownames(RNA)
file.h5[["ADT_names"]] <- rownames(ADT)
file.h5[["celltype.l1"]] <- pbmc_multimodal@meta.data$celltype.l1
file.h5[["celltype.l2"]] <- pbmc_multimodal@meta.data$celltype.l2
file.h5[["celltype.l3"]] <- pbmc_multimodal@meta.data$celltype.l3
file.h5$close_all()



############################################################
#
# DOGMA-seq, CITE-seq, and ASAP-seq PBMC
#
############################################################
pbmc <- readRDS('../data/pbmc_LLL_processed.rds')


DefaultAssay(pbmc) <- 'SCT'
pbmc <- FindVariableFeatures(pbmc, nfeatures=5000, assay='SCT')
RNA <- pbmc@assays$SCT@counts[VariableFeatures(pbmc),]
RNA <- RNA[apply(RNA>0, 1, sum)>=500,]
ADT <- pbmc@assays$ADT@counts
ADT <- ADT[apply(ADT>0, 1, sum)>=500,]
DefaultAssay(pbmc) <- 'peaks'
pbmc <- FindTopFeatures(pbmc, min.cutoff = 'q25')
peaks <- pbmc@assays$peaks@counts[VariableFeatures(pbmc),]
peaks <- peaks[apply(peaks>0, 1, sum)>=500,]
peaks <- peaks[!(startsWith(rownames(peaks), 'chrX') | 
                 startsWith(rownames(peaks), 'chrY')),]
peaks <- peaks[rownames(peaks)[order(match(rownames(peaks),rownames(pbmc)))],]

library(hdf5r)
file.h5 <- H5File$new("DOGMA_pbmc.h5", mode = "w")
file.h5[["ADT"]] <- as.matrix(ADT)

file.h5[["RNA.shape"]] <- RNA@Dim
file.h5[["RNA.data"]] <- RNA@x
file.h5[["RNA.indices"]] <- RNA@i
file.h5[["RNA.indptr"]] <- RNA@p

file.h5[["peaks.shape"]] <- peaks@Dim
file.h5[["peaks.data"]] <- peaks@x
file.h5[["peaks.indices"]] <- peaks@i
file.h5[["peaks.indptr"]] <- peaks@p

file.h5[["cell_ids"]] <- colnames(RNA)
file.h5[["gene_names"]] <- rownames(RNA)
file.h5[["ADT_names"]] <- rownames(ADT)
file.h5[["peak_names"]] <- rownames(peaks)
file.h5[["celltypes"]] <- as.character(pbmc@meta.data$wsnn_res.0.8)
file.h5[["predicted.celltypes"]] <- pbmc@meta.data$predicted.celltype.l1

file.h5[["batches"]] <- t(as.matrix(as.numeric(factor(pbmc@meta.data$stim)))-1)
file.h5$close_all()



coembed <- readRDS('data/22July2020_Seurat_Coembed4.rds')
citeseq <- subset(coembed, orig.ident=='RNA')

DefaultAssay(citeseq) <- "RNA"
citeseq <- SCTransform(citeseq, verbose = FALSE)

DefaultAssay(citeseq) <- 'SCT'
citeseq <- FindVariableFeatures(citeseq, nfeatures=5000, assay='SCT')
RNA <- citeseq@assays$SCT@counts[VariableFeatures(citeseq),]
RNA <- RNA[apply(RNA>0, 1, sum)>=250,]
ADT <- citeseq@assays$ADT@counts
ADT <- ADT[apply(ADT>0, 1, sum)>=250,]
cat(dim(RNA), dim(ADT))

file.h5 <- H5File$new("data/cite_pbmc.h5", mode = "w")
file.h5[["ADT"]] <- as.matrix(ADT)
file.h5[["RNA.shape"]] <- RNA@Dim
file.h5[["RNA.data"]] <- RNA@x
file.h5[["RNA.indices"]] <- RNA@i
file.h5[["RNA.indptr"]] <- RNA@p
file.h5[["cell_ids"]] <- colnames(RNA)
file.h5[["gene_names"]] <- rownames(RNA)
file.h5[["ADT_names"]] <- rownames(ADT)
celltypes <- rep('0', length(citeseq@meta.data$seurat_clusters))
celltypes[citeseq@meta.data$seurat_clusters %in% as.character(c(9))] <- "Mono"
celltypes[citeseq@meta.data$seurat_clusters %in% as.character(c(10))] <- "DC"
celltypes[citeseq@meta.data$seurat_clusters %in% as.character(c(4))] <- "B"
celltypes[citeseq@meta.data$seurat_clusters %in% as.character(c(3))] <- "NK"
celltypes[citeseq@meta.data$seurat_clusters %in% as.character(c(2,5))] <- "CD8 T"
celltypes[citeseq@meta.data$seurat_clusters %in% as.character(c(0,1,6,7,8,11))] <- "CD4 T"
file.h5[["celltypes"]] <- celltypes
file.h5[["batches"]] <- t(as.matrix(as.numeric(factor(citeseq@meta.data$stim)))-1)
file.h5$close_all()


asapseq <- subset(coembed, orig.ident=='ATAC')
cdf <- readRDS("data/ASAP_embedding_CLRadt.rds")
control_cells <- gsub("Control#", "", rownames(cdf)[cdf$sample == "Control"])
stim_cells <- gsub("Stim#", "", rownames(cdf)[cdf$sample == "Stim"])
pbmc <- readRDS('../data/pbmc_LLL_processed.rds')
DefaultAssay(pbmc) <- 'peaks'
frags.stim <- CreateFragmentObject(
  path = "data/stim_fragments.tsv.gz",
  cells = stim_cells
)
stim.counts <- FeatureMatrix(
  fragments = frags.stim,
  features = granges(pbmc),
  cells = stim_cells
)
frags.control <- CreateFragmentObject(
  path = "data/control_fragments.tsv.gz",
  cells = control_cells
)
control.counts <- FeatureMatrix(
  fragments = frags.control,
  features = granges(pbmc),
  cells = control_cells
)
asap_cells <- c(control_cells, stim_cells)
shared_cell <- asap_cells[asap_cells %in% colnames(atac_mat)]
peaks <- cbind(control.counts, stim.counts)
colnames(peaks) <- colnames(asapseq[['ADT']]@counts)
asapseq@meta.data$cellids <- colnames(peaks)
asapseq[["peaks"]] <- CreateChromatinAssay(
    counts = peaks,
    sep = c(":", "-"),
    #genome = 'hg38',
    #fragments = '../../../asap_large_data_files/multiome_pbmc_stim/input/fragments.tsv.gz',
    min.cells = 0,
    min.features = 0
)
DefaultAssay(asapseq) <- 'peaks'
asapseq <- FindTopFeatures(asapseq, min.cutoff = 'q25')
peaks <- asapseq@assays$peaks@counts[VariableFeatures(asapseq),]
peaks <- peaks[apply(peaks>0, 1, sum)>=250,]
peaks <- peaks[!(startsWith(rownames(peaks), 'chrX') | 
                 startsWith(rownames(peaks), 'chrY')),]
peaks <- peaks[rownames(peaks)[order(match(rownames(peaks),rownames(asapseq)))],]
ADT <- asapseq[['ADT']]@counts
ADT <- ADT[apply(ADT>0, 1, sum)>=250,]
cat(dim(ADT), dim(peaks))

file.h5 <- H5File$new("data/asap_pbmc.h5", mode = "w")
file.h5[["ADT"]] <- as.matrix(ADT)
file.h5[["peaks.shape"]] <- peaks@Dim
file.h5[["peaks.data"]] <- peaks@x
file.h5[["peaks.indices"]] <- peaks@i
file.h5[["peaks.indptr"]] <- peaks@p
file.h5[["cell_ids"]] <- colnames(ADT)
file.h5[["peak_names"]] <- rownames(peaks)
file.h5[["ADT_names"]] <- rownames(ADT)
celltypes <- rep('0', length(asapseq@meta.data$seurat_clusters))
celltypes[asapseq@meta.data$seurat_clusters %in% as.character(c(9))] <- "Mono"
celltypes[asapseq@meta.data$seurat_clusters %in% as.character(c(10))] <- "DC"
celltypes[asapseq@meta.data$seurat_clusters %in% as.character(c(4))] <- "B"
celltypes[asapseq@meta.data$seurat_clusters %in% as.character(c(3))] <- "NK"
celltypes[asapseq@meta.data$seurat_clusters %in% as.character(c(2,5))] <- "CD8 T"
celltypes[asapseq@meta.data$seurat_clusters %in% as.character(c(0,1,6,7,8,11))] <- "CD4 T"
file.h5[["celltypes"]] <- celltypes
file.h5[["batches"]] <- t(as.matrix(as.numeric(factor(asapseq@meta.data$stim)))-1)
file.h5$close_all()

