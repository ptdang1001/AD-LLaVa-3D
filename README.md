# AD-LLaVa-3D

# Predicting Alzheimer's Progression with a Multimodal Vision-Language Model

![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Coming%20Soon-yellow)
![Paper](https://img.shields.io/badge/Paper-Under%20Review-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Overview

This repository contains the code and resources for our research on predicting the future progression of Alzheimer's disease (AD). We leverage a state-of-the-art multimodal AI model to analyze both brain MRI scans and structured electronic health records (EHR) together, providing a more holistic and accurate forecast of how a patient's condition will evolve over time.

## 🧠 The Challenge: Why is this important?

Predicting how Alzheimer's disease will progress in a specific individual is incredibly challenging. The disease affects each person differently, and relying on just one type of data—like an MRI scan or a cognitive test—can miss the full picture. This work aims to provide patients, families, and clinicians with a better, more personalized tool for understanding future health outcomes.

## 🛠️ Our Approach: How does it work?

Our method combines the strengths of different data types by using a powerful vision-language model.

1.  **Foundation Model:** We start with **LLaVA-NeXT-Video**, a model that excels at understanding both images and text.
2.  **Analyzing MRI Scans:** We treat 3D MRI scans as "video-like" data. This allows the model to analyze the brain's structure slice by slice, understanding the full volumetric context rather than just a flat image.
3.  **Understanding Patient History:** Our key innovation is a method to convert sparse, tabular EHR data (like cognitive scores, demographics, and biomarkers) into rich, descriptive text prompts. This gives the model crucial clinical context to interpret the MRI scan.
4.  **Fine-Tuning for Prediction:** We then fine-tune the model on the public ADNI dataset, teaching it to connect the patterns in the MRI and EHR data from one point in time to predict clinical outcomes at a future visit.

## ✨ Key Features

* **Multimodal Analysis:** Integrates MRI (vision) and EHR (text) data for more accurate predictions.
* **Longitudinal Forecasting:** Predicts patient outcomes at various future time points (e.g., from visit 1 to visit 3, or visit 2 to visit 4).
* **Innovative Data Handling:** A novel strategy to create thousands of training samples from longitudinal patient records and handle sparse EHR data effectively.
* **State-of-the-Art Foundation:** Built on the powerful LLaVA-NeXT-Video architecture.

## 📈 Project Status & Future Plans

* **📜 Manuscript:** Our research paper detailing the full methodology and results is currently under peer review. We will post a link to the publication here once it has been accepted.
* **🤗 Hugging Face Model:** We plan to release the fine-tuned model weights on the Hugging Face Hub to allow the community to use and build upon our work. Stay tuned!
* **💻 Code:** The code in this repository will be further documented and cleaned up for easier use upon the paper's publication.

## 📝 Citation

If you find our work useful in your research, we would appreciate a citation to our paper once it is published.
