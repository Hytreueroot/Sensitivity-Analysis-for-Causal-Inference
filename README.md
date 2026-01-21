# Sensitivity-Analysis-for-Causal-Inference

[![Thesis](https://img.shields.io/badge/Read-Master%20Thesis-blue?style=for-the-badge&logo=adobeacrobatreader)](./Master_Thesis.pdf)
[![Python](https://img.shields.io/badge/Python-yellow?style=for-the-badge&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)

This repository contains the implementation, experiments, and datasets for my Master's Thesis: **"Sensitivity of Causal Effect Estimates Under Assumed Gaussian Noise to the True Latent Noise Distribution"**.

## 📄 Abstract
Estimating causal effects from observational data is frequently challenged by the presence of unobserved confounders. The $\rho$-GNF estimator addresses this challenge by using deep generative models to perform sensitivity analysis, parametrizing the latent confounding structure using a Gaussian copula. However, the reliability of this estimator when the true underlying latent distribution deviates from the Gaussian copula assumption remains a critical open question.

This thesis systematically evaluates the robustness of the $\rho$-GNF to misspecified latent structures. Utilizing a controlled simulation based strategy, we tested the model against data generating processes exhibiting non-linear (quadratic) and asymmetric tail-dependent (Clayton copula) latent structures. The analysis focused on whether the estimator could determine an "effective" sensitivity parameter that recovers the true Average Causal Effect (ACE) despite the distributional mismatch.

Our results demonstrate that the $\rho$-GNF is remarkably robust to structural misspecification. In both non-linear and tail-dependent scenarios, the model consistently yielded a specific sensitivity parameter that compensated for the bias, aligning the estimated ACE with the ground truth. Furthermore, the estimator exhibited high stability and monotonicity across repeated experiments. These findings suggest that the Gaussian copula serves as a flexible functional proxy for complex dependency structures, extending the practical applicability of the $\rho$-GNF beyond strictly Gaussian copula environments.

## 🔍 Key Findings & Discussion

Based on the empirical results discussed in the thesis, the study highlights three major conclusions regarding the $\rho$-GNF estimator:

### 1. Validity under Ideal Conditions
When the Data Generating Process (DGP) matches the structural assumptions (Gaussian latent structure), the estimator performs with high precision. The estimated sensitivity curves intersect the Ground Truth ACE exactly at the true correlation parameter ($\rho_{true}$), confirming the model's validity.

### 2. Robustness to Misspecification (The Compensatory Mechanism)
A critical finding of this research is the estimator's flexibility under **structural misspecification**. Even when the true latent dependency is **non-Gaussian** (tested via Quadratic and Clayton Copula DGPs):
* The model successfully recovers the true Average Causal Effect (ACE) within the valid parameter range.
* **Compensatory Mechanism:** The estimator approximates complex, unmodeled dependencies (such as tail asymmetry in Clayton or non-linearity in Quadratic) by shifting the effective linear correlation parameter ($\rho$). This suggests the model balances out the confounding bias by adjusting $\rho$, rather than failing to converge.

### 3. Stability
Despite the structural mismatch in stress tests, the estimator demonstrates high stability (indicated by narrow uncertainty bands) and preserves monotonicity. This suggests that the Gaussian copula can serve as a useful approximation for more complex dependency structures in causal estimation.

## 📂 Repository Structure

The project is modularized into specific packages for data generation, visualization, and the core model logic.

```text
├── Data_generation/          # Modules for synthetic data creation
│   ├── generate_gaussian_data.py       # Baseline Gaussian DGP
│   ├── generate_archimedean_data.py    # Clayton Copula (Tail Dependence)
│   └── generate_non_gaussian_data.py   # Quadratic/Non-linear DGP
│
├── Visualization/            # Plotting utilities
│   ├── plot_ace_curve.py               # Single ACE curve plotting
│   └── plot_mean_std_ace_curve.py      # Aggregated Mean/Std plots
│
├── sensitivity_model.py      # Core logic of the ρ-GNF estimator
├── Sensitivity_Analysis...ipynb # Main Jupyter Notebook for running experiments
└── Master_Thesis.pdf         # Full thesis documentation and mathematical proofs
