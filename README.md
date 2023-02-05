# GazeEstimation_alpha
<img src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white"/> <img src="https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white" />

Model for 3D Gaze Estimation based on [L2CS-Net](https://github.com/Ahmednull/L2CS-Net)

## Demo
comming soon...

## OverView
This is an appearance-based 3D gaze estimation method. It is a method to improve the accuracy of the conventional method, L2CS-Net.

To be presented at [IEICE](https://www.ieice-taikai.jp/2023general/jpn/) (as of 2023.02.02).


## Introduction
To improve the accuracy of the conventional method, L2CS-Net, we devised a method that uses face and both eye images as input. Our gaze estimation network is shown below. (These images are taken from Gaze360.)

![introfig](./pictures/basicmodel.png)


## Requirements
The project contains follwing files/folders.

- 'model.py' : the model code.
- 'train.py' : the entry for training and validation.
- 'test.py' : the entry fot testing.
- 'dataset.py' : the data loader code.
- 'utils.py' : the utils code.

## DataPreparing
* Dowanload [Gaze360](http://gaze360.csail.mit.edu/) dataset.

* Apply [pre-processing](http://phi-ai.buaa.edu.cn/Gazehub/3D-dataset/) to the dataset.

* The path of the dataset should be *./datasets/Gaze360*.
