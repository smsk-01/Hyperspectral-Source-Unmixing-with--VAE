# Hyperspectral Source Unmixing with β-VAE (Dirichlet Latent Space)

## 📌 Project Overview
This repository contains an implementation of **hyperspectral source unmixing** using a **β-Variational Autoencoder (β-VAE)** with a **Dirichlet latent prior** in **PyTorch**.  
The model decomposes each pixel spectrum into:
- **Endmembers** $E = [e_1, \dots, e_k]$ (pure spectral signatures),
- **Abundances** $a = [a_1, \dots, a_k]$ (fractions of materials).

The decoder is designed to recover the **linear mixing model** while also learning **nonlinear residuals**.

---

## 🔬 Hyperspectral Source Unmixing: Theory

### Linear Mixing Model (LMM)
A hyperspectral pixel $x \in \mathbb{R}^L$ can be modeled as:

$$
x = Ea + w = \sum_{i=1}^k a_i e_i + w
$$

with the **abundance constraints**:

$$
a_i \geq 0, \quad \sum_{i=1}^k a_i = 1
$$

- $E \in \mathbb{R}^{L \times k}$: endmember matrix,  
- $a \in \mathbb{R}^k$: abundance vector,  
- $w$: noise.  

---

### Nonlinear Mixing Extension
To better capture **nonlinear scattering effects**, we extend the model as:

$$
x \approx Ea + \phi(a)
$$

where $\phi(a)$ is a nonlinear correction learned by the decoder.  
This matches the implemented architecture where:

- $Ea$ is modeled by a **linear PositiveLinear layer** (`self.M`),
- $\phi(a)$ is modeled by a **deep decoder network** (`decoder_phi_a`).

---

### Dirichlet Latent Prior
Instead of Gaussian latents, abundances are drawn from a **Dirichlet distribution**:

$$
a \sim \text{Dirichlet}(\alpha), \quad \alpha = \text{Softmax}(f(x))
$$

This ensures that:
- $a_i \geq 0$ (positivity),
- $\sum_i a_i = 1$ (sum-to-one).  

Thus the VAE naturally enforces the **abundance constraints**.

---

### Loss Function
The training loss combines:
1. **Spectral Angle Distance (SAD)** — angular similarity between original and reconstructed spectra:

$$
\text{SAD}(x, \hat{x}) = \arccos \left( \frac{\langle x, \hat{x} \rangle}{\|x\|\|\hat{x}\|} \right)
$$

2. **MSE Reconstruction Loss** — penalizing intensity errors:

$$
\text{MSE}(x, \hat{x}) = \|x - \hat{x}\|^2
$$

3. **KL Divergence** between posterior $q(a|x)$ and Dirichlet prior $p(a)$:

$$
KL[q(a|x) \;||\; p(a)]
$$

4. Weighted β-VAE objective:

$$
\mathcal{L} = \text{SAD}^k + \gamma \, (\text{MSE})^m + \beta \, (KL)^n
$$

where $(k,m,n)$ are scaling exponents.

---

## 🛠️ Implementation Details
- **Framework**: [PyTorch](https://pytorch.org/)  
- **Encoder**: MLP with BatchNorm + ReLU  
- **Latent space**: Dirichlet via `Softmax` → `Dirichlet(alpha)`  
- **Decoder**:  
  - $Ea$ linear term via `PositiveLinear` (`self.M`)  
  - Nonlinear correction $\phi(a)$ via deep MLP (`decoder_phi_a`)  
- **Loss components**: SAD, MSE, Dirichlet KL divergence  
- **Dataset**: Samson (3 endmembers)  

---

## 📊 Results
- Extracted **endmembers** close to ground-truth spectra,  
- Produced **abundance maps** with smooth spatial structure,  
- Outperformed classical NMF/VCA under noise and nonlinearity.
- <img width="847" height="547" alt="image" src="https://github.com/user-attachments/assets/b9ffa839-7e92-4381-97e1-188debe042d8" />
---

## 📂 Repository Structure
