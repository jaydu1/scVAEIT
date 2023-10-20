Sys.setenv("OMP_NUM_THREADS" = 4)
Sys.setenv("OPENBLAS_NUM_THREADS" = 4)
Sys.setenv("MKL_NUM_THREADS" = 6)
Sys.setenv("VECLIB_MAXIMUM_THREADS" = 4)
Sys.setenv("NUMEXPR_NUM_THREADS" = 6)

library(Seurat)
library(SeuratDisk)
library(Signac)

library(dplyr)
library(harmony)


##############################################################################
#
# Load data
#
##############################################################################

# CITE-seq
load("result/ex2/dogma_cite.RData")
DefaultAssay(citeseq) <- 'SCT'
citeseq <- FindVariableFeatures(citeseq, nfeatures=5000, assay='SCT')
RNA <- citeseq@assays$SCT@counts[VariableFeatures(citeseq),]
RNA <- RNA[apply(RNA>0, 1, sum)>=250,]
ADT <- citeseq@assays$ADT@counts
ADT <- ADT[apply(ADT>0, 1, sum)>=250,]
celltypes <- rep('0', length(citeseq@meta.data$seurat_clusters))
celltypes[citeseq@meta.data$seurat_clusters %in% as.character(c(9))] <- "Mono"
celltypes[citeseq@meta.data$seurat_clusters %in% as.character(c(10))] <- "DC"
celltypes[citeseq@meta.data$seurat_clusters %in% as.character(c(4))] <- "B"
celltypes[citeseq@meta.data$seurat_clusters %in% as.character(c(3))] <- "NK"
celltypes[citeseq@meta.data$seurat_clusters %in% as.character(c(2,5))] <- "CD8 T"
celltypes[citeseq@meta.data$seurat_clusters %in% as.character(c(0,1,6,7,8,11))] <- "CD4 T"

data_ref_cite <- CreateSeuratObject(RNA)
data_ref_cite[["ADT"]] <- CreateAssayObject(ADT, assay='ADT')
data_ref_cite@meta.data$stim <- citeseq@meta.data$stim

DefaultAssay(data_ref_cite) <- "ADT"
data_ref_cite <- data_ref_cite  %>% 
  NormalizeData(assay = "ADT", normalization.method = "CLR", margin = 2) %>% FindVariableFeatures()


# ASAP-seq
load("result/ex2/dogma_asap.RData")
DefaultAssay(asapseq) <- 'peaks'
asapseq <- FindTopFeatures(asapseq, min.cutoff = 'q25')
peaks <- asapseq@assays$peaks@counts[VariableFeatures(asapseq),]
peaks <- peaks[apply(peaks>0, 1, sum)>=250,]
peaks <- peaks[!(startsWith(rownames(peaks), 'chrX') | 
                 startsWith(rownames(peaks), 'chrY')),]
peaks <- peaks[rownames(peaks)[order(match(rownames(peaks),rownames(asapseq)))],]

ADT <- asapseq[['ADT']]@counts
ADT <- ADT[apply(ADT>0, 1, sum)>=250,]

celltypes <- rep('0', length(asapseq@meta.data$seurat_clusters))
celltypes[asapseq@meta.data$seurat_clusters %in% as.character(c(9))] <- "Mono"
celltypes[asapseq@meta.data$seurat_clusters %in% as.character(c(10))] <- "DC"
celltypes[asapseq@meta.data$seurat_clusters %in% as.character(c(4))] <- "B"
celltypes[asapseq@meta.data$seurat_clusters %in% as.character(c(3))] <- "NK"
celltypes[asapseq@meta.data$seurat_clusters %in% as.character(c(2,5))] <- "CD8 T"
celltypes[asapseq@meta.data$seurat_clusters %in% as.character(c(0,1,6,7,8,11))] <- "CD4 T"

data_ref_asap <- CreateSeuratObject(ADT, assay='ADT')
data_ref_asap[['peaks']] <- CreateChromatinAssay(peaks)
data_ref_asap@meta.data$stim <- asapseq@meta.data$stim

DefaultAssay(data_ref_asap) <- "ADT"
data_ref_asap <- data_ref_asap  %>% 
  NormalizeData(assay = "ADT", normalization.method = "CLR", margin = 2) %>% FindVariableFeatures()


# DOGMA-seq
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

data_ref_dogma <- CreateSeuratObject(RNA)
data_ref_dogma[["ADT"]] <- CreateAssayObject(ADT, assay='ADT')
data_ref_dogma[['peaks']] <- CreateChromatinAssay(peaks)
data_ref_dogma@meta.data$stim <- pbmc@meta.data$stim



##############################################################################
#
# Harmony + 3-WNN
#
##############################################################################

run_integrate <- function(data_ref, knn.range=200, k.nn=20){
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
                                        dims.list = list(1:30, 2:30, 1:30),
                                          knn.range=knn.range, k.nn=k.nn
                                       )
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
    data_ref
}

data_ref_dogma <- run_integrate(data_ref_dogma, 200)


##############################################################################
#
# Map query
#
##############################################################################
map_query <- function(data_ref, data_query){
    anchors <- FindTransferAnchors(
        reference = data_ref,
        query = data_query,
        k.filter = NA,
        reference.reduction = "spca", 
        reference.neighbors = "spca.annoy.neighbors", 
        dims = 1:50
    )

    data_query <- MapQuery(
        anchorset = anchors, 
        query = data_query,
        reference = data_ref, 
        refdata = 'ADT',
        reference.reduction = "spca",
        reduction.model = "wnn.3.umap"
    )
    
    data_query
}


data_ref_cite <- map_query(data_ref_dogma, data_ref_cite)
data_ref_asap <- map_query(data_ref_dogma, data_ref_asap)

umap_coord <- rbind(
    data_ref_dogma[['wnn.3.umap']]@cell.embeddings,
    data_ref_cite[['ref.umap']]@cell.embeddings,
    data_ref_asap[['ref.umap']]@cell.embeddings
)
write.csv(umap_coord[,1:2], 'result/ex3/full/Seurat_embedding.csv')
save.image(file = "result/ex3/dogma_combined.RData")
