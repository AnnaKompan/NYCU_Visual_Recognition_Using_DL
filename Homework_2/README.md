# NYCU Visual Recognition Using Deep Learning 2025 Spring HW2

StudentID: 313540009
Name: Anna Kompan (安娜)

## Introduction

For this task I've used

- ResNet50 as the model backbone,removed global average pooling and fully connected layer.
- Anchor generator for detecting objects
- Roi pooling converts regions proposed by RPN into output (feature map)
- Faster RCNN model with 11 classes (10 for digits and 1 for background)

For Task 1 - Target is corresponding class and bounding box of each image.
For Task 2 - Target is classified number on image.

Dataset consists of training, validation and testing folders, also train and valid json files in COCO format

## How to install

Need to install dependencies:

```
pip install torch torchvision
pip install tqdm
pip install pandas
pip install Pillow
pip install pycocotools
```

Recommend using Conda for version cotrol
Python version used is 3.11.11

## Performance snapshot

![Performance snapshot](/Performance_snapshot.png)
