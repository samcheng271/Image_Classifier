# CSE 144 Applied Machine Learning Final Project

Date: 2023/12/10

Member: 
1. Samuel Cheng(1772680) samachen@ucsc.edu
2. Craig Hunter(1505840) craig.p.hunter@gmail.com
3. Junya Ihira(1929535) jihira@ucsc.edu


## Overview

The objective of this project is to predict labels for a given image dataset. Each team member trained a different model and obtained the label probability, and we combined all of them through ensemble methods to generate the final predictions.

## Task distribution

1. Samuel Cheng worked on the Swin Transformer model
2. Craig Hunter implemented ResNet 152
3. Junya Ihira focused on ViT-H/14 and the implementation of the ensemble method.

## Directory structure

```bash
.
|-- README.md 
|-- data
|   |-- processed # Stores label probabilities
|   |-- result # Holds the final predictions
|   |-- raw # Contains raw data
|-- program
|   |-- craig
|   |   |-- README-craig.md
|   |-- sam
|   |   |-- README-sam.md
|   |-- junya
|   |   |-- README-junya.md  ...
|   |-- ensemble # Manages files related to ensemble methods
|       |-- main.ipynb 

```


## Installation
To run this project, follow the steps below:

1. Obtain label probabilities from each model

Execute the code in the respective folders under ```program/<name>```, collect the label probabilities, and save the results in ```data/processed/```. Please refer to the respective READMEs for specific instructions on how to run each code to get the label probabilities.

2. Generate final predictions

Run ```program/ensemble/main.ipynb```. The final predictions will be saved as ```data/result/result.csv```. We achieved a Kaggle score of 0.89 in the end.


