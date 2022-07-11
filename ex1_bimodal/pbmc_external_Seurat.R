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



############################################################
#
# Preprocess reference data, and run WNN and sPCA
#
############################################################
data_ref <- CreateSeuratObject(counts = RNA, project = "pbmc")
data_ref[["ADT"]] <- CreateAssayObject(counts = ADT)

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

save(data_ref, file = sprintf('full_ref.RData'))
load(sprintf('full_ref.RData'))
save.image(file = sprintf("pbmc_full_ref.RData"))

# load(file = sprintf("pbmc_full_ref.RData"))


############################################################
#
# Impute RNA of the hold-out celltype
#
############################################################

for(query in c('cbmc', 'reap'){
    if(query=='cbmc'){

        cbmc <- LoadH5Seurat("data/cbmc.h5seurat")
        RNA <- cbmc@assays$RNA@counts
        ADT <- cbmc@assays$ADT@counts
        ADT <- ADT[,colSums(RNA)>=200]
        RNA <- RNA[,colSums(RNA)>=200]
        RNA <- RNA[rowSums(RNA)>=10,]
        RNA <- RNA[!startsWith(rownames(RNA), 'MOUSE'),]
        ADT_names <- rownames(ADT)
        ADT_names[ADT_names=='CD3'] <- 'CD3-1'
        ADT_names[ADT_names=='CD4'] <- 'CD4-2'
        ADT_names[ADT_names=='CD56'] <- 'CD56-1'
        rownames(ADT) <- ADT_names
    }else{
        reap_rna_2 <- read.table("data/REAPseq/GSM2685238_mRNA_2_PBMCs_matrix.txt.gz", 
            row.names = 1)
        reap_rna_3 <- read.table("data/REAPseq/GSM2685239_mRNA_3_PBMCs_matrix.txt.gz", 
            row.names = 1)
        reap_protein_2 <- read.table("data/REAPseq/GSM2685243_protein_2_PBMCs_matrix.txt.gz", 
            row.names = 1)
        reap_protein_3 <- read.table("data/REAPseq/GSM2685244_protein_3_PBMCs_matrix.txt.gz", 
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

        reap <- CreateSeuratObject(RNA)
        adt_assay <- CreateAssayObject(counts = ADT)
        reap[["ADT"]] <- adt_assay
        reap[["percent.mt"]] <- PercentageFeatureSet(reap, pattern = "^MT-")
        reap <- subset(reap, subset = nFeature_RNA >= 200 & percent.mt < 5)

        ADT_names <- rownames(reap@assays$ADT)
        ADT_names[ADT_names=='CD11b'] <- 'CD11b_1'
        ADT_names[ADT_names=='CD45'] <- 'CD45_2'
        ADT_names[ADT_names=='CD56'] <- 'CD56_1'
        ADT_names[ADT_names=='CD158E1'] <- 'CD158e1'
        ADT_names[ADT_names=='CD4'] <- 'CD4_2'
        ADT_names[ADT_names=='CD4.1'] <- 'CD4_1'
        ADT_names[ADT_names=='CD3'] <- 'CD3_2'
        ADT_names[ADT_names=='CD8'] <- 'CD8.2'
        ADT_names[ADT_names=='CD8.1'] <- 'CD8'

        RNA <- reap@assays$RNA@counts
        ADT <- reap@assays$ADT@counts
        rownames(ADT) <- ADT_names
    }


    ############################################################
    #
    # Impute RNA
    #
    ############################################################
    data_query <- CreateSeuratObject(counts = ADT, project = "test", assay = "ADT")

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
        reduction.model = "wnn.umap",
        transferdata.args = list(k.weight=40)
      )


    data_RNA <- CreateSeuratObject(counts = RNA, project = "cbmc", assay = "RNA")
    VariableFeatures(data_RNA) <- rownames(data_RNA[["RNA"]])
    data_RNA <- NormalizeData(data_RNA)

    rownames(data_RNA@assays$RNA@data) <- toupper(gsub("\\.", "-", gsub("_", "-", 
        rownames(data_RNA@assays$RNA@data))))
    rownames(data_query@assays$pred_RNA@data) <- toupper(gsub("\\.", "-", gsub("_", "-", 
        rownames(data_query@assays$pred_RNA@data))))                                  
    gene_names <- intersect(rownames(data_RNA@assays$RNA@data), rownames(data_query@assays$pred_RNA@data))
    gene_names <- sort(gene_names) # 3464 for cbmc and 3864 for reap
    X_test <- as.matrix(data_RNA@assays$RNA@data[gene_names,])
    X_hat <- as.matrix(data_query@assays$pred_RNA@data[gene_names,])
        
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
      sprintf('result/%s/res_Seurat_RNA.csv', query), 
      quote=FALSE)

    rm(data_RNA)
    rm(data_query)
                 
    ############################################################
    #
    # Impute ADT
    #
    ############################################################
 
    data_query <- CreateSeuratObject(counts = RNA, project = "cbmc_test")
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


    data_ADT <- CreateSeuratObject(counts = ADT, project = "cbmc", assay = "ADT")
    VariableFeatures(data_ADT) <- rownames(data_ADT[["ADT"]])
    if(trans=='CLR'){
        data_ADT <- NormalizeData(data_ADT, normalization.method = trans, margin = 2)
    }else{
        data_ADT <- NormalizeData(data_ADT, normalization.method = trans)
    }


    shared_ADT <- sort(intersect(
        rownames(data_ADT@assays$ADT@data), 
        rownames(data_query@assays$pred_ADT@data)
    )) # 10 for cbmc and 38 for reap
    size_factor <- apply(
        data_ADT@assays$ADT@counts[shared_ADT,], 2, function(x){
            exp(x = sum(log1p(x = x[x > 0]), na.rm = TRUE) / length(x = x))}
    )
                 
    Y_test <- apply(data_ADT@assays$ADT@counts[shared_ADT,], 2, function(xx){log(xx/sum(xx)*10000 + 1)})
    Y_hat <- data_query@assays$pred_ADT@data[shared_ADT,]         
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
      sprintf('result/%s/res_Seurat_ADT.csv', query), 
      quote=FALSE)
    
    rm(data_ADT)
    rm(data_query)
                 
}



