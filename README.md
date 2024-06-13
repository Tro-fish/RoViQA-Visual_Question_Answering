<div align="center">

# (Open-ended) Visual-Question Answering

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
<br><b> Visual-Question-Answering model created by combining Roberta and ViT</b> <br>
This repository contains the code for a Visual Question Answering (VQA) model. The model combines image features and text features extracted using ViT and RoBERTa. The project includes training, inference, and various utility scripts.
</div>

## Model Architecture
<p align="center">
  <img src="https://github.com/Tro-fish/Visual-Question-Answering/assets/79634774/f7d0eb20-f3b4-4f69-880d-d412ed32ab68" alt="Description of the image" width="100%" />
</p>

## Table of Contents
- [Existing Code](#existing-code)
- [Training Details and Evaluation Results](#training-details-and-evaluation-results)
- [How to Train](#how-to-train)
- [How to Perform Inference](#how-to-perform-inference)
- [Hyperparameters](#hyperparameters)
- [Model Parameters](#model-parameters)
- [Machine Environment](#machine-environment)

## Existing Code

- **vqa_model.py**: Definition of the VQA model architecture.
- **vqa_dataset.py**: Script to handle the dataset loading and preprocessing.
- **train.py**: Script to train the VQA model.
- **inference.py**: Script to perform inference using the trained VQA model.
- **parser.py**: Utility script to parse command-line arguments for training and inference.
