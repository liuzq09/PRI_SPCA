Code for spectral initialization of sparse phase retrieval using sparse principal component analysis methods (PRI-SPCA). 
The codes are mainly adapted from the codes for the paper: https://arxiv.org/abs/1705.06412, shared at https://github.com/GauriJagatap/model-copram

Please cite the following paper:

Zhaoqiang Liu, Subhroshekhar Ghosh, and Jonathan Scarlett, "Towards Sample-Optimal Compressive Phase Retrieval with Sparse and Generative Priors." Accepted to Conference on Neural Information Processing Systems (NeurIPS), 2021.
The arXiv version is available at: https://arxiv.org/abs/2106.15358

Main codes:
demo_main.m: This script is for reproducing our figures (Figures 1,2,3) presented in the main paper.
demo_supp.m: This script is for reproducing our figures (Figures 4,5,6) presented in the supplementary material.

Runs and compares performances of the following sparse phase retrieval algorithms:
1. Compressive phase retrieval with alternating minimization (CoPRAM)
(implemented based on the paper https://arxiv.org/abs/1705.06412)
2. PRI-SPCA
3. PRI-SPCA-NT
4. Thresholded Wirtinger Flow for sparse phase retrieval (ThWF)
(implemented based on the paper https://arxiv.org/abs/1506.03382)
5. Sparse Phase Retrieval using Truncated Amplitude Flow (SPARTA)
(implemented based on the paper https://arxiv.org/abs/1611.07641)
6. Random initialization 




