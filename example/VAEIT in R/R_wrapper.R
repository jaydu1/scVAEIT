# import Python package (call after setting up python environment with reticulate)
# Use 'reticulate' to access Python environment
library(reticulate)
use_condaenv(condaenv = "tf")
# If the package is installed via PyPI in the Python environment,
# simply import it:
scVAEIT <- import("scVAEIT")
# Otherwise, need to downloaded Github repo, set a proper path to import the module.
# setwd('..')
# scVAEIT <- import("scVAEIT")


##################################################################
#
# VAE
#
##################################################################
#' run VAE
#'
#' @param data An n-by-p input matrix containing missing entries, where n and p are the number 
#'      of cells and features, respectively.
#' @param masks An n-by-p binary matrix indicating the missing entries.
#' @param batches_cate A n-by-b1 matrix containing categorical/binary covariates.
#' @param batches_cont A n-by-b2 matrix containing continuous covariates.
#' @param conditions A n-by-b3 matrix containing ordinal covariates. Compared to batches effects, these are stronger effects to be removed.
#' @param num_epoch The number of training epoches.
#' @param print_every_epoch A p-by-n initialization matrix.
#' @param return_mean whether return the mean imputed matrix or the multiple imputed matrices.
#' @param seed The random seed.
vae <- function(
    # data
  data, masks, batches_cate, batches_cont, conditions=NULL,
  # network structure
  dim_input_arr=NULL, dimensions=c(16), dim_latent=4L,
  dim_block=NULL, dist_block=NULL, dim_block_enc=c(64), dim_block_dec=c(64),
  block_names=NULL, uni_block_names=NULL, dim_block_embed=128L, skip_conn=FALSE,
  # hyperparameters for training
  num_epoch=20L, print_every_epoch=8L,
  beta_kl=1., beta_unobs=0.9, beta_reverse=0., beta_modal=NULL, p_feat=0.5, p_modal=NULL, seed=0,
  model_dir='r_checkpoint/', verbose=TRUE, return_mean=TRUE){
  
  data <- as.matrix(data)
  
  config = list(
    # A network structure of 
    # x     :              dim_input -> 64 -> 16 -> z 4 -> 16 -> 64 -> dim_input
    #                                 |                  |
    # masks : dim_input -> dim_embed ->                 ->
    'dim_input_arr'=dim_input_arr,
    'dimensions'=dimensions, # hidden layers
    'dim_latent'=dim_latent, # latent space
    
    # block structures
    'dist_block'=dist_block,
    'dim_block'=dim_block,
    
    'dim_block_enc'=dim_block_enc,
    'dim_block_dec'=dim_block_dec,
    'block_names'=block_names,
    'uni_block_names'=uni_block_names,
    'dim_block_embed'=dim_block_embed,
    
    # some hyperparameters
    'beta_kl'=beta_kl,
    'beta_unobs'=beta_unobs,
    'beta_modal'=beta_modal,
    'beta_reverse'=beta_reverse,
    
    # prob of random maskings
    "p_feat"=p_feat,
    "p_modal"=p_modal,
    
    'skip_conn'=skip_conn
  )
  scVAEIT$reset_random_seeds(as.integer(seed))
  
  cat('Initializing model...\n')
  model <- scVAEIT$VAEIT(config, data, masks, NULL, batches_cate, batches_cont, conditions)
  
  cat('Training model...\n')
  model$train(
    num_epoch=as.integer(num_epoch), # the number of iterations, generally ~10 would be 
    # good. If this is too large, it may overfit the data.
    # if X is provided, then evaluate the model every 4 epochs
    checkpoint_dir=NULL,
    save_every_epoch=as.integer(print_every_epoch),
    verbose=verbose
  )
  model$save_model(model_dir)
  
  X_imp <- model$get_denoised_data(return_mean=return_mean)
  
  res <- list()
  res <- process_output(res, return_mean, X_imp, data, 'X_imp')
  
  X_blend <- data
  ina <- masks!=0
  X_blend[ina] <- res[['X_imp']][ina]
  rownames(X_blend) <- rownames(data)
  colnames(X_blend) <- colnames(data)
  res[['X_blend']] <- X_blend
  
  return(res)
}


process_output <- function(res, return_mean, X_imp, data, name){
  if(return_mean){
    X_imp <- X_imp
  }else{
    X_imp <- aperm(X_imp, c(2,1,3))
    dimnames(X_imp)[[2]] <- rownames(data)
    dimnames(X_imp)[[3]] <- colnames(data)
    res[['X_samples']] =X_imp
    X_imp <- colMeans(X_imp, dims=1)
  }
  
  rownames(X_imp) <- rownames(data)
  colnames(X_imp) <- colnames(data)
  res[[name]] <- X_imp
  res
}