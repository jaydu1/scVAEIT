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

library(dplyr)
library(harmony)


##############################################################################
#
# Prepare data
#
##############################################################################
pbmc <- readRDS('data/pbmc_LLL_processed.rds')
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
stim <- pbmc@meta.data$stim
celltypes <- pbmc@meta.data$predicted.celltype.l1


data_ref <- CreateSeuratObject(RNA)
data_ref[["ADT"]] <- CreateAssayObject(ADT, assay='ADT')
data_ref[['peaks']] <- CreateChromatinAssay(peaks)
data_ref@meta.data$stim <- stim



# RNA Seurat stuff
DefaultAssay(data_ref) <- "RNA"
data_ref <- data_ref  %>% 
  NormalizeData() %>% ScaleData() %>%
  FindVariableFeatures() %>%
  RunPCA(verbose = FALSE, assay = "RNA", reduction.name = "pca") %>%
  RunHarmony( group.by.vars = 'stim', reduction = 'pca', assay.use = 'RNA',
             project.dim = FALSE,  reduction.save = "harmony_RNA")

# LSI dim reduction
DefaultAssay(data_ref) <- "peaks"
data_ref <- RunTFIDF(data_ref) %>% 
  FindTopFeatures(min.cutoff = NULL) %>%
  RunSVD() %>%
  RunHarmony( group.by.vars = 'stim', reduction = 'lsi', assay.use = 'peaks',
             project.dim = FALSE,  reduction.save = "harmony_Peaks")

# Do it for ADT
DefaultAssay(data_ref) <- "ADT"
data_ref <- data_ref  %>% 
  NormalizeData(assay = "ADT", normalization.method = "CLR", margin = 2) %>%
  ScaleData(assay = "ADT", do.scale = FALSE) %>%
  FindVariableFeatures(assay = "ADT") %>% 
  RunPCA(verbose = FALSE, assay = "ADT", reduction.name = 'apca') %>%
  RunHarmony( group.by.vars = 'stim', reduction = 'apca', assay.use = 'ADT',
             project.dim = FALSE,  reduction.save = "harmony_ADT")


data_ref <- FindMultiModalNeighbors(object = data_ref,
                                    reduction.list = list("harmony_RNA", "harmony_Peaks", "harmony_ADT"),
                                    dims.list = list(1:30, 2:30, 1:30))
data_ref <- RunSPCA(data_ref, assay = 'RNA', graph = 'wsnn')
data_ref <- RunUMAP(data_ref, nn.name = "weighted.nn",
                    reduction.name = "wnn.3.umap", reduction.key = "Uw3_" , return.model = TRUE)

data_ref <- FindNeighbors(
  object = data_ref,
  reduction = "spca",
  dims = 1:50,
  graph.name = "spca.annoy.neighbors", 
  k.param = 50,
  cache.index = TRUE,
  return.neighbor = TRUE,
  l2.norm = TRUE
)
dir.create(sprintf("result/ex2/full/Seurat/"), showWarnings = FALSE, recursive = TRUE)
save(data_ref, file = sprintf('result/ex2/full/Seurat/dogma_full_ref.RData'))
save.image(file = sprintf("result/ex2/full/Seurat/dogma_ref.RData"))

load(file = sprintf("result/ex2/full/Seurat/dogma_ref.RData"))


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
save(citeseq, file = "result/ex2/dogma_cite.RData")



asapseq <- subset(coembed, orig.ident=='ATAC')
cdf <- readRDS("data/ASAP_embedding_CLRadt.rds")
control_cells <- gsub("Control#", "", rownames(cdf)[cdf$sample == "Control"])
stim_cells <- gsub("Stim#", "", rownames(cdf)[cdf$sample == "Stim"])
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
save(asapseq, file = "result/ex2/dogma_asap.RData")


##############################################################################
#
# Integration
#
##############################################################################


load("result/ex2/dogma_cite.RData")
load("result/ex2/dogma_asap.RData")
load("result/ex2/dogma_ref_full.RData")

DefaultAssay(data_ref) <- "ADT"
DefaultAssay(asapseq) <- "ADT"
asapseq <- asapseq  %>% 
  NormalizeData(assay = "ADT", normalization.method = "CLR", margin = 2) %>% FindVariableFeatures()

DefaultAssay(citeseq) <- "ADT"
citeseq <- citeseq  %>% 
  NormalizeData(assay = "ADT", normalization.method = "CLR", margin = 2) %>% FindVariableFeatures()
data.list <- list('dogma'=data_ref, 'cite'=citeseq, 'asap'=asapseq)
features <- SelectIntegrationFeatures(object.list = data.list)
anchors <- FindIntegrationAnchors(object.list = data.list, anchor.features = features)
combined <- IntegrateData(anchorset = anchors)

DefaultAssay(combined) <- "integrated"

# Run the standard workflow for visualization and clustering
combined <- ScaleData(combined, verbose = FALSE)
combined <- RunPCA(combined, npcs = 30, verbose = FALSE)
combined <- RunUMAP(combined, reduction = "pca", dims = 1:30)
combined <- FindNeighbors(combined, reduction = "pca", dims = 1:30)
combined <- FindClusters(combined, resolution = 0.5)

write.csv(combined[["umap"]]@cell.embeddings, 'Seurat_umap.csv')
save(combined, file = "result/ex3/dogma_combined.RData")

