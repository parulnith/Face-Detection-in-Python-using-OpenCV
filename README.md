# Face Detection with Python using OpenCV
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/parulnith/Face-Detection-in-Python-using-OpenCV/master)
![](https://github.com/parulnith/Face-Detection-in-Python-using-OpenCV/blob/master/data/pic.png)

Face detection is a computer vision technology that helps to locate/visualize human faces in digital images. This technique is a specific use case of object detection technology that deals with detecting instances of semantic objects of a certain class (such as humans, buildings or cars) in digital images and videos. With the advent of technology, face detection has gained a lot of importance especially in fields like photography, security, and marketing.

## Objective
This is the repository linked to the tutorial with the same name. The idea is to introduce people to the concept of object detection in Python using the OpenCV library and how it can be utilized to perform tasks like Facial detection.

## Blogpost
[Face Detection with Python using OpenCV](https://levelup.gitconnected.com/face-detection-with-python-using-opencv-5c27e521c19a)

## Installation
OpenCV-Python supports all the leading platforms like Mac OS, Linux, and Windows. It can be installed in either of the following ways:

**1. From pre-built binaries and source :**

Please refer to the detailed documentation here for Windows and here for Mac.

**2. [Unofficial pre-built OpenCV packages for Python](https://pypi.org/project/opencv-python/).**

Packages for standard desktop environments (Windows, macOS, almost any GNU/Linux distribution)

run ```pip install opencv-python``` if you need only the main modules
run ```pip install opencv-contrib-python``` if you need both main and contrib modules (check extra modules listing from [OpenCV documentation](https://docs.opencv.org/master/))

## Table Of Contents

* [Images as Arrays](#images-as-arrays)
* Images and OpenCV
* Basic Operations on Images
* Face Detection


### Images as Arrays

An image is nothing but a standard Numpy array containing pixels of data points. More the number of pixels in an image, the better is its resolution. You can think of pixels to be tiny blocks of information arranged in form a 2 D grid and the depth of a pixel refers to the colour information present in it. In order to be  processed by a computer, an image needs to be converted into a binary form. The colour of an image can be calculated as follows:

      Number of colours/ shades = 2^bpp where bpp represents bits per pixel.

Naturally, more the number of bits/pixels , more possible colours in the images. The following table shows  the relationship more clearly. 

![](https://github.com/parulnith/Face-Detection-in-Python-using-OpenCV/blob/master/Images%20as%20Arrays/Bits%20per%20pixels.png)
