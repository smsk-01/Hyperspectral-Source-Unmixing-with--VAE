# Hyperspectral Source Unmixing with β-VAE

## Project Overview
This repository contains an implementation of **hyperspectral source unmixing** using a **β-Variational Autoencoder (β-VAE)** in **PyTorch**.  
The objective is to decompose a hyperspectral image into a set of pure spectral signatures (*endmembers*) and their corresponding *abundance maps*.  

Hyperspectral unmixing has numerous applications in **remote sensing**, **geosciences**, **environmental monitoring**, and **agriculture**, where it is crucial to detect and quantify the materials present in an observed scene.

---

## Hyperspectral Source Unmixing: Theory
A hyperspectral image can be seen as a 3D data cube:  

- Each pixel corresponds to a **spectrum** (a vector of reflectance values over many wavelengths).  
- Each spectrum is generally a **linear mixture** of pure material spectra, called **endmembers**.  
- The problem of **spectral unmixing** consists in retrieving:  
  - The **endmembers** (pure sources),  
  - The **abundance maps** (fractions of each material in each pixel).  

Mathematically, for each pixel $x$:  
$x$ $\approx$ $\sum_{i=1}^{k}$ $a_i$ $e_i$, with $a_i$ $\geq$ $0$ and $\sum_i a_i$ $=$ $1$



where:  
- $e_i$ = endmembers (pure spectral signatures)  
- $a_i\$ = abundances (proportions of materials) 

---

## Classical Approaches (without AI)
Before deep learning, hyperspectral unmixing was tackled with **geometric and optimization-based methods**, such as:

- **PCA / NMF (Non-Negative Matrix Factorization)** – factorizing spectra under positivity constraints  
- **VCA (Vertex Component Analysis)** – extracting convex vertices in spectral space  
- **Sparse regression-based methods** – enforcing sparsity in abundance estimation  

These methods are interpretable but often limited when noise, nonlinear mixing, or complex spatial structures are present.

---

## Deep Learning & Autoencoders
Autoencoders have emerged as a powerful tool for hyperspectral unmixing:

- **Standard Autoencoders** learn a compressed latent representation of spectra and reconstruct them  
- **Variational Autoencoders (VAEs)** add a **probabilistic latent space**, improving **regularization** and allowing **robust generation**  
- **β-VAE** introduces a scaling factor β on the KL-divergence term, encouraging **disentangled latent factors**, which is beneficial for separating spectral sources  

### Latent Space Constraints
Two approaches for enforcing abundance constraints are explored:

1. **Truncated Gaussian latent space** – reparameterization + normalization  
2. **Dirichlet latent space** – ensures non-negativity and sum-to-one naturally  

---

## 🛠️ Implementation Details
- **Framework**: [PyTorch](https://pytorch.org/)  
- **Model**: β-VAE with customizable encoder/decoder architectures  
- **Dataset**: Samson dataset (3 endmembers)  
- **Evaluation metrics**:
  - **MSE** between original and reconstructed spectra  
  - **Spectral Angle Mapper (SAM)** for endmember similarity  
  - **Spatial coherence** of abundance maps  

---

## 📊 Results & Analysis
The trained β-VAE model can:  

- Extract **endmembers** consistent with ground-truth reference spectra  
- Provide **abundance maps** that are spatially coherent  
- Outperform classical baselines in terms of robustness to noise and reconstruction error  

> ⚠️ A critical discussion of results is included in the notebook, highlighting trade-offs between different latent space parameterizations (Gaussian vs Dirichlet) and the role of β in disentanglement.

---

## 📂 Repository Structure
