# NYCU Computer Vision 2025 Spring HW1

StudentID: 313540009
Name: Anna Kompan (安娜)

## Introduction

For this task I've used ResNet50 as the model backbone to classify images correctly (Mollusca, Chordata, Arthropoda, Echinodermata, Tracheophyta, Bryophyta, Basidiomycota, Ascomycota).
Target label is corresponding object category id of the image.
Dataset consists of training, validation and testing folders. In total there are 100 categories and corresponding 100 (0 to 9) folders.

## How to install

Need to install dependencies:

```
pip install torch torchvision
pip install numpy
pip install pandas
pip install matplotlib
pip install scikit-learn
```

Recommend using Conda for version cotrol
Python version 3.12.7

## Performance snapshot

![Performance snapshot](/Performance_snapshot.png)
