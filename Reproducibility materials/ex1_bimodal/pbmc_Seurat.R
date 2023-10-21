Sys.setenv("OMP_NUM_THREADS" = 4)
Sys.setenv("OPENBLAS_NUM_THREADS" = 4)
Sys.setenv("MKL_NUM_THREADS" = 6)
Sys.setenv("VECLIB_MAXIMUM_THREADS" = 4)
Sys.setenv("NUMEXPR_NUM_THREADS" = 6)

library(Seurat)
library(SeuratDisk)
library(Signac)

library(dplyr)

pbmc_multimodal <- LoadH5Seurat("data/pbmc_multimodal.h5seurat", assays = "counts", reductions = FALSE, graphs = FALSE,)

RNA <- pbmc_multimodal@assays$SCT@counts[VariableFeatures(pbmc_multimodal),]
RNA <- RNA[rowSums(RNA>0)>=500,]
ADT <- pbmc_multimodal@assays$ADT@counts
ADT <- ADT[rowSums(ADT>0)>=500,]

celltypes <- pbmc_multimodal@meta.data$celltype.l1

rm(pbmc_multimodal)


trans <- 'CLR'
cell_type_list <- c('Mono', 'CD4 T')
args <- commandArgs(trailingOnly = TRUE)
cell_type_test <- cell_type_list[as.numeric(args[1])+1L]
cat(cell_type_test)

data_ADT <- CreateSeuratObject(counts = ADT[,celltypes==cell_type_test], project = "pbmc", assay = "ADT")
Y_test <- as.matrix(apply(data_ADT@assays$ADT@counts, 2, function(xx){log(xx/sum(xx)*10000 + 1)}))
data_RNA <- CreateSeuratObject(counts = RNA[,celltypes==cell_type_test], project = "pbmc", assay = "RNA")
data_RNA <- NormalizeData(data_RNA)
X_test <- as.matrix(data_RNA@assays$RNA@data)

size_factor <- apply(
    data_ADT@assays$ADT@counts, 2, function(x){
        exp(x = sum(log1p(x = x[x > 0]), na.rm = TRUE) / length(x = x))}
)
rm(data_ADT, data_RNA)



############################################################
#
# Preprocess reference data, and run WNN and sPCA
#
############################################################
data_ref <- CreateSeuratObject(counts = RNA[,celltypes!=cell_type_test], project = "pbmc")
data_ref[["ADT"]] <- CreateAssayObject(counts = ADT[,celltypes!=cell_type_test])

DefaultAssay(data_ref) <- 'RNA'
data_ref <- NormalizeData(data_ref) %>% FindVariableFeatures() %>% ScaleData() %>% RunPCA()

DefaultAssay(data_ref) <- 'ADT'
VariableFeatures(data_ref) <- rownames(data_ref[["ADT"]])
if(trans=='CLR'){
    data_ref <- NormalizeData(data_ref, normalization.method = trans, margin = 2) %>% 
        ScaleData() %>% RunPCA(reduction.name = 'apca')
}else{
    data_ref <- NormalizeData(data_ref, normalization.method = trans) %>% 
        ScaleData() %>% RunPCA(reduction.name = 'apca')
}


data_ref <- FindMultiModalNeighbors(
  data_ref, reduction.list = list("pca", "apca"), 
  dims.list = list(1:30, 1:18), modality.weight.name = "RNA.weight"
)
data_ref <- RunSPCA(data_ref, assay = 'RNA', graph = 'wsnn')

data_ref <- RunUMAP(data_ref, nn.name = "weighted.nn", reduction.name = "wnn.umap", 
              reduction.key = "wnnUMAP_", return.model = TRUE)

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

dir.create(sprintf("result/ex1/%s/Seurat/", cell_type_test), showWarnings = FALSE, recursive = TRUE)
save(data_ref, file = sprintf('result/ex1/%s/Seurat/%s_ref.RData',, cell_type_test, cell_type_test))
# load(sprintf('result/ex1/%s/Seurat/%s_ref.RData', cell_type_test, cell_type_test))
save.image(file = sprintf("result/ex1/%s/Seurat/pbmc_%s_ref.RData", cell_type_test, cell_type_test))
# load(file = sprintf("result/ex1/%s/Seurat/pbmc_%s_ref.RData", cell_type_test, cell_type_test))


############################################################
#
# Impute RNA of the hold-out celltype
#
############################################################

data_query <- CreateSeuratObject(counts = ADT[,celltypes==cell_type_test], project = "pbmc_test", assay = "ADT")
if(trans=='CLR'){
    data_query <- NormalizeData(data_query, normalization.method = trans, margin = 2) %>% FindVariableFeatures() %>% 
        ScaleData() %>% RunPCA()
}else{
    data_query <- NormalizeData(data_query, normalization.method = trans) %>% FindVariableFeatures() %>% 
        ScaleData() %>% RunPCA()
}

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
    refdata = list(pred_RNA = "RNA"),
    reference.reduction = "spca",
    reduction.model = "wnn.umap"
  )

X_hat <- as.matrix(data_query@assays$pred_RNA@data)
rs <- sapply(1:dim(X_hat)[1], function(i) cor.test(x=X_test[i,], y=X_hat[i,], method = 'pearson')$estimate[['cor']])
ss <- sapply(1:dim(X_hat)[1], function(i) cor.test(x=X_test[i,], y=X_hat[i,], method = 'spearman', exact=FALSE)$estimate[['rho']])
mse <- apply((X_test - X_hat)^2, 1, mean)
res_details <- data.frame(
    'RNA'=rownames(X_hat),
    'Pearson r'=rs, 
    'Spearman r'=ss,
    'MSE'=mse, check.names = FALSE,
    stringsAsFactors=FALSE)
write.csv(res_details, 
  sprintf('result/ex1/%s/Seurat/res_Seurat_RNA.csv', cell_type_test), 
  quote=FALSE)

res_overall <- data.frame(
    Source=rep(NA, 2),
    Target=rep("", 2),
    'Pearson r'=rep(NA, 2), 
    'Spearman r'=rep(NA, 2),
    'MSE'=rep(NA, 2), check.names = FALSE,
    stringsAsFactors=FALSE)             
X_hat_ <- as.vector(X_hat)
X_test_ <- as.vector(X_test)
res_overall[1, ] <- c('ADT', 'RNA', 
      cor.test(x=X_test_, y=X_hat_, method = 'pearson')$estimate[['cor']],
      cor.test(x=X_test_, y=X_hat_, method = 'spearman', exact=FALSE)$estimate[['rho']],
      mean((X_test_-X_hat_)**2))


############################################################
#
# Impute ADT of the hold-out celltype
#
############################################################
if('data_query' %in% ls()){rm(data_query)}
data_query <- CreateSeuratObject(counts = RNA[,celltypes==cell_type_test], project = "pbmc_test")
data_query <- NormalizeData(data_query) %>% FindVariableFeatures() %>% ScaleData() %>% RunPCA()

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
    refdata = list(pred_ADT = "ADT"),
    reference.reduction = "spca",
    reduction.model = "wnn.umap"
  )


Y_hat <- data_query@assays$pred_ADT@data            
Y_hat <- sweep(exp(Y_hat) - 1, MARGIN=2, size_factor, `*`)
Y_hat <- apply(Y_hat, 2, function(xx){log(xx/sum(xx)*10000 + 1)})
Y_hat <- as.matrix(Y_hat)
rs <- sapply(1:dim(Y_hat)[1], function(i) cor.test(x=Y_test[i,], y=Y_hat[i,], method = 'pearson')$estimate[['cor']])
ss <- sapply(1:dim(Y_hat)[1], function(i) cor.test(x=Y_test[i,], y=Y_hat[i,], method = 'spearman', exact=FALSE)$estimate[['rho']])
mse <- apply((Y_test - Y_hat)^2, 1, mean)
res_details <- data.frame(
    'ADT'=rownames(Y_hat),
    'Pearson r'=rs, 
    'Spearman r'=ss,
    'MSE'=mse, check.names = FALSE,
    stringsAsFactors=FALSE)
write.csv(res_details, 
  sprintf('result/ex1/%s/Seurat/res_Seurat_ADT.csv', cell_type_test), 
  quote=FALSE)

Y_hat_ <- as.vector(Y_hat)
Y_test_ <- as.vector(Y_test)
res_overall[2, ] <- c('RNA', 'ADT', 
      cor.test(x=Y_test_, y=Y_hat_, method = 'pearson')$estimate[['cor']],
      cor.test(x=Y_test_, y=Y_hat_, method = 'spearman', exact=FALSE)$estimate[['rho']],
      mean((Y_test_-Y_hat_)**2))
write.csv(res_overall, 
  sprintf('result/ex1/%s/Seurat/res_Seurat_overall.csv', cell_type_test), 
  quote=FALSE)

