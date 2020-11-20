# ShowMeMore 

## Description

A simple image recognition written in python, which works by training on some build-in image dataset and test some preselected images.

## How it works

I've chosen an image dataset from **Keras** which contains over  60.000 images in 10 classes with 6000 images per class. Also, there are 50.000 training images and 10.000 test images.

By creating the model, and compiling it. By compiling it, it defines the loss function, the optimizer/learning rate and the metrics. When the model is called it predicts the labels of the imported images and then it prints the labels with the prediction accuracy.

## How to run 

**Steps**

1. Run `git clone https://github.com/george951/ShowMeMore.git`
2. Download [Python3](https://www.python.org/downloads/)
3. Download [Pip](https://pip.pypa.io/en/stable/installing/)
4. Download [Tensorflow](https://www.tensorflow.org/install)
5. Run  `python3 -m pip install -U matplotlib` 
6. Run  `pip install numpy`
7. Run  `pip install glob`
8. Run  `pip install scikit-image`

To run the program press `python3 main.py`
