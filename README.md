# DynammoVI: Dynamic Multi-omic Variational Inference for somatic evolution within Single Cell Data (in-Dev)

## Overview

DynammoVI is a prototype computational framework (in-Dev) designed to integrate single-cell multi-omic data with a focus on capturing the dynamics of somatic evolution and the complexities of aging. By leveraging variational autoencoders (VAEs), DynammoVI aims to construct a comprehensive landscape of cell states, taking into account the stochastic nature of somatic mutations and their implications on epigenetic and gene expression profiles across different tissues.

Inspired by the principles of MultiVI, inVAE, and looking forward to the possibilities with SAMSVAE, DynammoVI introduces a novel approach to model the temporal non-stationarity in single-cell data, providing insights into the relationships between mutational complexity, aging, and cellular function.

## Model Architecture

DynammoVI features an advanced architecture with distinct encoders for scRNA-seq, ATAC-seq, and SNV data, each meticulously tailored to the specific characteristics of these data types. The model is engineered to produce two unique latent spaces: a variant latent space influenced by covariates such as age and an invariant latent space that enables counterfactual analysis, drawing inspiration from inVAE.

### Encoders

- **scRNA-seq and ATAC-seq Encoders**: These encoders employ a zero-inflated negative binomial distribution to effectively model the count data, accounting for dropout events and overdispersion commonly observed in single-cell omics.

  $$p(x|z) = \text{ZINB}(\mu(z), \theta(z), \pi(z))$$

  Here, $( \mu_(z) \)$ denotes the mean, $( \theta(z) \)$ the dispersion, and $( \pi(z) \)$ the dropout probability, all parameterized by the latent variable $( z \)$.

- **SNV Encoder**: This encoder uses a Bernoulli distribution for the binary SNV data, incorporating a Gumbel-Softmax trick to facilitate gradient-based optimization, offering a differentiable approximation for discrete data modeling.

  $$p(x|z) = \text{Bernoulli}(\sigma(z))$$

  In this context, $( \sigma(z) $) represents the mutation presence probability, parameterized by the latent variable $( z $).

### Decoders

DynammoVI includes dedicated decoders for each data modality, ensuring the reconstructed data maintains the original statistical properties. These decoders map the latent representations back to the data space, enabling the model to learn a generative process for each modality.

### Adversarial Network

An adversarial component is seamlessly integrated to promote invariance in the latent space concerning specific covariates like age, facilitating counterfactual scenario exploration and enhancing latent representation interpretability.

## Objectives and Loss Functions

Training DynammoVI entails optimizing a composite loss function that encompasses:

- **Reconstruction Losses**: Tailored for each data modality to ensure precise input data reconstruction from the latent representations.
- **KL Divergence**: Aims to regularize the latent space to adhere to a predefined distribution, typically Gaussian, fostering a smooth latent space.
- **Adversarial Loss**: Designed to diminish the adversarial network's capacity to predict the covariate from the invariant latent space, thereby reinforcing invariance.

## Inspiration and Future Directions

DynammoVI is inspired by:

- **MultiVI**: For its comprehensive approach to multi-omic single-cell data modeling.
- **inVAE**: For its innovative use of invariant representations to facilitate counterfactual analysis in single-cell data.
- **SAMSVAE**: Envisioned as a future direction to accommodate temporally non-stationary data, capturing the dynamic essence of cellular processes and somatic evolution.

## Goal

DynammoVI's primary objective is to offer an exhaustive framework that not only clarifies the intricate interplay between genetic mutations, epigenetic modifications, and gene expression dynamics but also considers the temporal dynamics of cellular aging and somatic evolution. By achieving this, DynammoVI aims to enrich our understanding of cellular functionality and disease progression at the single-cell level, unveiling new insights into aging mechanisms and the somatic evolution of cells across various tissues.

## Installation and Usage

DynammoVI is currently under development and is available for installation directly from the repository:

```bash
pip install git+https://github.com/Issacgoh/BreadcrumbsscDynammoVI
```

**Note**: As DynammoVI is in the prototype stage, features and functionalities are subject to change. Users are encouraged to check back regularly for updates and enhancements.

## Project Team

- Issac Goh, Newcastle University; Sanger Institute ([https://haniffalab.com/team/issac-goh.html](https://haniffalab.com/team/issac-goh.html))

## Contact

For inquiries, feedback, or contributions, please contact Issac Goh at

 [ig7@sanger.ac.uk](mailto:ig7@sanger.ac.uk).

## Built With

- Scanpy
- PyTorch
- Additional libraries and packages to be listed

## Getting Started

To utilize DynammoVI, you will need:

- An Anndata object containing your single-cell data.
- A categorical variable within the Anndata object to represent labels or states.

## Running Locally

DynammoVI is designed to be used within a Jupyter notebook environment to take full advantage of interactive display interfaces. Functions can also be executed locally through a Python script. For usage examples, please refer to the notebook provided in "/example_notebooks/".

## Production

For deploying DynammoVI on high-performance computing clusters or cloud virtual machines, especially for large-scale data analysis, please consult the example configurations and scripts available in "/example_notebooks/". or consult the SCENTINEL framework for deriving multi-modal meta cells.
