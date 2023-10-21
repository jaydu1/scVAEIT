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

cell_type_list <- c('CD4 T', 'CD8 T', 'B', 'NK', 'DC', 'Mono')
args <- commandArgs(trailingOnly = TRUE)
cell_type_test <- cell_type_list[as.numeric(args[1])+1L]
cat(cell_type_test)

data_ref <- CreateSeuratObject(RNA[,celltypes!=cell_type_test])
data_ref[["ADT"]] <- CreateAssayObject(ADT[,celltypes!=cell_type_test], assay='ADT')
data_ref[['peaks']] <- CreateChromatinAssay(peaks[,celltypes!=cell_type_test])
data_ref@meta.data$stim <- stim[celltypes!=cell_type_test]



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

batch_test <- as.numeric(factor(stim))[celltypes==cell_type_test] - 1
dir.create(sprintf("result/ex2/%s/Seurat/", cell_type_test), showWarnings = FALSE, recursive = TRUE)
save(data_ref, file = sprintf('result/ex2/%s/Seurat/dogma_%s_ref.RData', cell_type_test, cell_type_test))
save.image(file = sprintf("result/ex2/%s/Seurat/dogma_ref.RData", cell_type_test))


############################################################
#
# Test on the hold-out cell type
#
############################################################

library(reticulate)
# need to change the path of python
use_python("anaconda3/envs/py/bin/python")
sklearn <- import("sklearn")
log_loss <- sklearn$metrics$log_loss
auc <- sklearn$metrics$roc_auc_score

n_modal <- 3
res <- data.frame(
    Target=rep("", n_modal*2),
    Batch=rep(NA, n_modal*2),
    'Metric 1'=rep(NA, n_modal*2), 
    'Metric 2'=rep(NA, n_modal*2),
    'MSE'=rep(NA, n_modal*2), check.names = FALSE,
    stringsAsFactors=FALSE)


modality_list <- c('RNA','ADT','peaks')
for(i in c(1:3)){
    modality <- modality_list[i]
    DefaultAssay(data_ref) <- "RNA"
    
    data_query <- CreateSeuratObject(RNA[,celltypes==cell_type_test])
    data_query[["ADT"]] <- CreateAssayObject(ADT[,celltypes==cell_type_test], assay='ADT')
    data_query[['peaks']] <- CreateChromatinAssay(peaks[,celltypes==cell_type_test])
    data_query@meta.data$stim <- stim[celltypes==cell_type_test]

    DefaultAssay(data_query) <- "RNA"
    data_query <- data_query  %>% 
      NormalizeData() %>% ScaleData() %>%
      FindVariableFeatures()
    DefaultAssay(data_query) <- "ADT"
    data_query <- data_query  %>% 
      NormalizeData(assay = "ADT", normalization.method = "CLR", margin = 2)
    DefaultAssay(data_query) <- "peaks"    
    data_query <- RunTFIDF(data_query) %>% 
      FindTopFeatures( min.cutoff = 'q0') %>%
      RunSVD()
    DefaultAssay(data_query) <- "RNA"
    DefaultAssay(data_ref) <- "RNA"
    
    refdata <- modality
    if(modality=='ADT'){
        size_factor <- apply(
            data_query[[modality]]@counts, 2, function(x){
                exp(x = sum(log1p(x = x[x > 0]), na.rm = TRUE) / length(x = x))}
        )
        x <- apply(data_query[[modality]]@counts, 2, function(xx){log(xx/sum(xx)*10000 + 1)})
    } else if(modality=='RNA'){
        x <- data_query[[modality]]@data
        DefaultAssay(data_ref) <- "ADT"
        DefaultAssay(data_query) <- "ADT"
    } else{ 
        x <- data_query[[modality]]@counts
        x <- x[!(startsWith(rownames(x), 'chrX') | startsWith(rownames(x), 'chrY')),]
        refdata <- GetAssayData(data_ref, assay = "peaks", slot = "counts")
        refdata[refdata>0.] <- 1.
        x[x>0.] <- 1.
    }
    data_query[[modality]] <- NULL
    
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
        refdata = list(pred = refdata),
        reference.reduction = "spca",
        reduction.model = "wnn.3.umap"
    )
    
    
    y <- data_query[['pred']]@data
    if(modality=='ADT'){
        y <- sweep(exp(y) - 1, MARGIN=2, size_factor, `*`)        
        y <- apply(y, 2, function(xx){log(xx/sum(xx)*10000 + 1)})
    }
    
    x <- as.matrix(x)
    y <- as.matrix(y)
    
    if((modality=='ADT') & (cell_type_test=='CD4T')){
        write.csv(y, sprintf('result/ex2/%s/Seurat/Seurat_ADT_pred.csv', cell_type_test), quote=FALSE)
    }
    
    for(batch in c(0,1)){
        if(modality!='peaks'){
            rs <- sapply(1:dim(y)[1], function(i) cor.test(
                x=x[i,][batch_test==batch], y=y[i,][batch_test==batch], method = 'pearson')$estimate[['cor']])
            ss <- sapply(1:dim(y)[1], function(i) cor.test(
                x=x[i,][batch_test==batch], y=y[i,][batch_test==batch], method = 'spearman', exact=FALSE)$estimate[['rho']])
            mse <- apply((x[,batch_test==batch] - y[,batch_test==batch])^2, 1, mean)
            res_details <- data.frame(
                'ADT'=rownames(y),
                'Pearson r'=rs, 
                'Spearman r'=ss,
                'MSE'=mse, check.names = FALSE,
                stringsAsFactors=FALSE)
            write.csv(res_details, 
                sprintf('result/ex2/%s/Seurat/res_Seurat_%s_%d.csv', cell_type_test, modality, batch),
                quote=FALSE)
        }else{
            aucs <- sapply(1:dim(x)[1], function(i){if(any(x[i,][batch_test==batch]==1)){
                return(auc(x[i,][batch_test==batch], y[i,][batch_test==batch]))}else{return(NA)}})
            bces <- sapply(1:dim(x)[1], function(i){if(any(x[i,][batch_test==batch]==1)){
                return(log_loss(x[i,][batch_test==batch], y[i,][batch_test==batch]))}else{return(NA)}})       
            mse <- apply((x[,batch_test==batch] - y[,batch_test==batch])^2, 1, mean)
            res_details <- data.frame(
                'ATAC'=rownames(y),
                'AUC'=aucs, 
                'BCE'=bces,
                'MSE'=mse, check.names = FALSE,
                stringsAsFactors=FALSE)
            
            write.csv(res_details, 
                sprintf('result/ex2/%s/Seurat/res_Seurat_%s_%d.csv', cell_type_test, 'ATAC', batch), 
                quote=FALSE)
        }
    }
    

    for(batch in c(0,1)){
        x_ <- as.vector(x[,batch_test==batch])
        y_ <- as.vector(y[,batch_test==batch]) 
        if(modality!='peaks'){
            res[2*(i-1)+batch+1, ] <- c(modality, batch,
                cor.test(x=x_, y=y_, method = 'pearson')$estimate[['cor']],
                cor.test(x=x_, y=y_, method = 'spearman', exact=FALSE)$estimate[['rho']],
                mean((x_-y_)**2))
        }else{
            res[2*(i-1)+batch+1, ] <- c('ATAC', batch,
                auc(x_, y_),
                log_loss(x_, y_),
                mean((x_-y_)**2))
        }
    }

}

write.csv(res, 
    sprintf('result/ex2/%s/Seurat/res_Seurat_overall.csv', cell_type_test), 
    quote=FALSE)
      

############################################################
#
# Test on the hold-out cell type with random missing
#
############################################################
mask_list_all <- read.csv('data/mask_dogma.csv', header=TRUE, row.names=1)
col_names <- colnames(mask_list_all)
col_names[c(1:nrow(RNA))] <- rownames(data_ref@assays$RNA)
col_names[c((nrow(RNA)+1):(nrow(RNA)+nrow(ADT)))] <- rownames(data_ref@assays$ADT)
col_names[-c(1:(nrow(RNA)+nrow(ADT)))] <- rownames(data_ref@assays$peaks)
colnames(mask_list_all) <- col_names

n_modal <- 3
args <- commandArgs(trailingOnly = TRUE)
cat(args)
for(ip in c(1:8)){
    
    n_simu <- 10
    p <- ip/10
    mask_list <- mask_list_all[(n_simu*(ip-1)+1):(n_simu*ip),]
    res <- data.frame(
        i=rep(NA, n_modal*n_simu*2),
        Batch=rep(NA, n_modal*n_simu*2),
        Target=rep("", n_modal*n_simu*2),
        'Metric 1'=rep(NA, n_modal*n_simu*2), 
        'Metric 2'=rep(NA, n_modal*n_simu*2),
        'MSE'=rep(NA, n_modal*n_simu*2), check.names = FALSE,
        stringsAsFactors=FALSE)

    

    for(i in 1:n_simu){
        if('data_query' %in% ls()){rm(data_query)}
        
        data_query <- CreateSeuratObject(RNA[
            rownames(data_ref@assays$RNA)[mask_list[i, rownames(data_ref@assays$RNA)] == 0],
            celltypes==cell_type_test])
        data_query[["ADT"]] <- CreateAssayObject(ADT[
            rownames(data_ref@assays$ADT)[mask_list[i, rownames(data_ref@assays$ADT)] == 0],
            celltypes==cell_type_test], assay='ADT')
        data_query[['peaks']] <- CreateChromatinAssay(peaks[
            rownames(data_ref@assays$peaks)[mask_list[i, rownames(data_ref@assays$peaks)] == 0],
            celltypes==cell_type_test])
        data_query@meta.data$stim <- stim[celltypes==cell_type_test]

        DefaultAssay(data_query) <- "RNA"
        data_query <- data_query  %>% 
          NormalizeData() %>% ScaleData() %>%
          FindVariableFeatures()
        DefaultAssay(data_query) <- "ADT"
        data_query <- data_query  %>% 
          NormalizeData(assay = "ADT", normalization.method = "CLR", margin = 2)
        DefaultAssay(data_query) <- "peaks"    
        data_query <- RunTFIDF(data_query) %>% 
          FindTopFeatures( min.cutoff = 'q0') %>%
          RunSVD()
        DefaultAssay(data_query) <- "RNA"

        anchors <- FindTransferAnchors(
            reference = data_ref,
            query = data_query,
            k.filter = NA,
            reference.reduction = "spca", 
            reference.neighbors = "spca.annoy.neighbors", 
            dims = 1:50
        )
        
        tryCatch({
            data_query <- MapQuery(
                anchorset = anchors, 
                query = data_query,
                reference = data_ref, 
                refdata = list(pred_ADT = "ADT", pred_RNA = "RNA", pred_peaks = peaks[,celltypes!=cell_type_test]),
                reference.reduction = "spca",
                reduction.model = "wnn.3.umap"
              )
            
            X_hat <- data_query@assays$pred_RNA@data
            X_hat <- as.matrix(X_hat[rownames(data_ref@assays$RNA)[mask_list[i, rownames(data_ref@assays$RNA)] == 1],])
            X_test_sub <- X_test[rownames(data_ref@assays$RNA)[mask_list[i, rownames(data_ref@assays$RNA)] == 1],]
            
            Y_hat <- data_query@assays$pred_ADT@data            
            Y_hat <- sweep(exp(Y_hat) - 1, MARGIN=2, size_factor, `*`)
            Y_hat <- apply(Y_hat, 2, function(xx){log(xx/sum(xx)*10000 + 1)})
            Y_hat <- as.matrix(Y_hat[rownames(data_ref@assays$ADT)[mask_list[i, rownames(data_ref@assays$ADT)] == 1],])
            Y_test_sub <- Y_test[rownames(data_ref@assays$ADT)[mask_list[i, rownames(data_ref@assays$ADT)] == 1],]
            
            Z_hat <- data_query@assays$pred_peaks@data
            Z_hat <- as.matrix(Z_hat[rownames(data_ref@assays$peaks)[mask_list[i, rownames(data_ref@assays$peaks)] == 1],])
            Z_test_sub <- Z_test[rownames(data_ref@assays$peaks)[mask_list[i, rownames(data_ref@assays$peaks)] == 1],]

            for(batch in c(0:1)){
                X_test_ <- as.vector(X_test_sub[,batch_test==batch])
                X_hat_ <- as.vector(X_hat[,batch_test==batch])
                res[6*(i-1)+1+3*batch, ] <- c(i, batch, 'RNA', 
                            cor.test(x=X_test_, y=X_hat_, method = 'pearson')$estimate[['cor']],
                            cor.test(x=X_test_, y=X_hat_, method = 'spearman', exact=FALSE)$estimate[['rho']],
                            mean((X_test_-X_hat_)**2))
                
                Y_test_ <- as.vector(Y_test_sub[,batch_test==batch])
                Y_hat_ <- as.vector(Y_hat[,batch_test==batch])
                res[6*(i-1)+2+3*batch, ] <- c(i, batch, 'ADT', 
                            cor.test(x=Y_test_, y=Y_hat_, method = 'pearson')$estimate[['cor']],
                            cor.test(x=Y_test_, y=Y_hat_, method = 'spearman', exact=FALSE)$estimate[['rho']],
                            mean((Y_test_-Y_hat_)**2)
                                                )
            
                Z_test_ <- as.vector(Z_test_sub[,batch_test==batch])
                Z_hat_ <- as.vector(Z_hat[,batch_test==batch])
                res[6*(i-1)+3+3*batch, ] <- c(i, batch, 'ATAC', 
                            auc(Z_test_, Z_hat_),
                            log_loss(Z_test_, Z_hat_),
                            mean((Z_test_-Z_hat_)**2)
                                                )
                
                }
            },
            error = function(e) {
                message(sprintf('Caught an error (i=%d)!', i))
                message(e)
            }
        )           
        write.csv(res, 
        sprintf('result/ex2/%s/Seurat/res_Seurat_masked_%.01f_%d.csv', cell_type_test, p, i), 
        quote=FALSE)             
    }

    write.csv(res, 
        sprintf('result/ex2/%s/Seurat/res_Seurat_masked_%.01f.csv', cell_type_test, p), 
        quote=FALSE)
}
