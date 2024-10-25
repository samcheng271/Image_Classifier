
# CSE 144 Applied Machine Learning Final Project Junya's work


Date: December 10, 2023

Name: Junya Ihira

UCSC ID: 1929535

For any questions or suggestions, feel free to reach out to us at:
Email: jihira@ucsc.edu

## Overview

The purpose of this program is to predict the label probability in a given image dataset using Vision Transformer-H/14 (ViT-H/14). I utilized the pre-trained ViT-H/14 available in PyTorch and fine-tuned it to make predictions on specific data.



## Directory structure
```bash
.
├── scripts/
│   ├── main.ipynb # Main Jupyter Notebook
│   └── module/
│       └── function.py # Script defining functions used as modules
├── data/
│   ├── raw/ # Raw datasets (train and test dataset)
│   ├── processed/ # Processed data
│   ├── result/ # Storage for output files (prediction, probability table, and loss history file)
│   └── figure/ # Storage for graphs (loss and accuracy figures)
├── models/ # Storage for fine-tuned models
├── environment.txt # List of dependencies
├── README.md # Project description and usage instructions
└── .gitignore # Git ignore settings file

```

## Installation

To run this project, follow the steps below:


1. Install Dependencies
Please create a conda environment and activate it.


```bash
conda create --name torch-junya python==3.9
conda activate torch-junya
```

```bash
pip install numpy pandas matplotlib
conda install -c pytorch pytorch
conda install -c pytorch torchvision
```

2. Move to the directory

```bash
cd junya
```

3. Prepare for Training and Test Data
Place the training and test data in ```data/raw/``` as follows:


```bash
data/raw/train
data/raw/test
```

4. Prepare for the fine-tuned model weights

Download the fine-tuned model weights from my Google Drive:

[Download Model Weights](https://drive.google.com/file/d/1zsthFuX1wYuSZJfv1vQRjsHx5AT_cjls/view?usp=sharing)

Place the weights in the models directory:
```bash
models/vit_h_14_fine_tuned40.pth
```


5. Open Jupyter notebook and run the code

Use ```scripts/main.ipynb``` and follow the instructions in it. You can get the label probability as ```data/processed/vit_h14_output.csv```. 


