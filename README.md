# 🧠 VAE-MULTIMODEL
### Multi-Modal Synthetic Data Generation Using Variational Autoencoders

<p align="center">

  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-ML-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />

</p>

<p align="center">

  <img src="https://img.shields.io/badge/Generative%20AI-VAE-8A2BE2?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Synthetic%20Data-Generation-00A86B?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Deep%20Learning-Research-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-Under%20Development-orange?style=for-the-badge" />

</p>

---

## 🌟 Project Overview

**VAE-MULTIMODEL** is a multi-modal deep generative framework designed to generate synthetic data using **Variational Autoencoders (VAEs)**.

The framework explores VAE-based synthetic data generation across three different data modalities:

| Modality | Dataset | Model | Output |
|---|---|---|---|
| 🖼️ Image | MVTec AD – Bottle | Image VAE | Synthetic Images |
| 📊 Tabular / CSV | Customer Churn | Tabular VAE | Synthetic CSV Data |
| 💬 Text | SMS Spam Collection | Text VAE | Synthetic Text |

The main objective is to learn meaningful latent representations of different types of data and use those representations to generate new synthetic samples.

---

# 🎯 Objectives

The project aims to:

- 🧠 Understand the architecture of Variational Autoencoders.
- 🔐 Explore synthetic data generation for privacy-preserving applications.
- 🖼️ Generate synthetic image data.
- 📊 Generate synthetic tabular/CSV data.
- 💬 Generate synthetic text data.
- 🔬 Compare VAE behavior across different data modalities.
- 📈 Evaluate reconstruction and generation quality.
- 🧩 Build a modular framework that can be extended to additional datasets.

---

# 🏗️ System Architecture

```text
                         ┌──────────────────────────┐
                         │       INPUT DATA         │
                         └────────────┬─────────────┘
                                      │
             ┌────────────────────────┼────────────────────────┐
             │                        │                        │
             ▼                        ▼                        ▼
       🖼️ IMAGE DATA             📊 CSV DATA              💬 TEXT DATA
        MVTec AD               Customer Churn          SMS Spam Dataset
             │                        │                        │
             ▼                        ▼                        ▼
      Image Preprocessing      Tabular Preprocessing    Text Preprocessing
             │                        │                        │
             ▼                        ▼                        ▼
        VAE Encoder             VAE Encoder             VAE Encoder
             │                        │                        │
             └───────────────┬────────┴────────┬───────────────┘
                             │
                             ▼
                    🧠 LATENT REPRESENTATION
                             │
                      Reparameterization
                             │
                             ▼
                         VAE Decoder
                             │
             ┌───────────────┼────────────────┐
             │               │                │
             ▼               ▼                ▼
       Synthetic Images  Synthetic CSV   Synthetic Text
