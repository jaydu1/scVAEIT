.. causarray documentation master file, created by
   sphinx-quickstart on Mon Jan 13 17:38:13 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Contents:
=======================

*scVAEIT* (Variational autoencoder for multimodal single-cell mosaic integration and transfer learning) is a Python module for multi-omic data integration and imputation, designed to handle various data types such as scRNA-seq, scATAC-seq, and proteomics.

.. toctree::
   :maxdepth: 1
   :caption: Introduction

   readme_link.md


.. toctree::
   :maxdepth: 1
   :glob:
   :caption: Main functions

   main_function/VAEIT
   main_function/prepare_data_input.ipynb
   main_function/parameter.ipynb   

.. toctree::
   :maxdepth: 1
   :glob:
   :caption: Tutorials (Python)

   tutorial/python/imputation_1modality.ipynb
   tutorial/python/imputation_2modalities.ipynb
   tutorial/python/integration_3modalities.ipynb

.. toctree::
   :maxdepth: 1
   :glob:
   :caption: Tutorials (R)

   tutorial/R/imputation_scRNAseq.ipynb
   tutorial/R/imputation_peptide.ipynb
