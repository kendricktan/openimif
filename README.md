# openimif (Open-soruced image identifer)
Open-sourced tool that uses opencv and tensorflow to 'learn' and identify stuff from images

![digit extraction](http://i.imgur.com/SQ3HECb.png)

# How?
- OpenCV is used to manipulate the properties of the image
- Tensorflow is used for the learning of 'elements' within the image

# What can it do? (currently)
- v1.0
> Able to extract digits out of an image

# Todo
1. Fix contour regions (don't scan for the contour within the zero)
2. Increase accuracy for digit recognition

# Getting-started
### Install Dependencies
For ubuntu:
Refer to http://www.pyimagesearch.com/2015/07/20/install-opencv-3-0-and-python-3-4-on-ubuntu/ to install OpenCV3, and numpy on your system

Refer to https://www.tensorflow.org/versions/master/get_started/os_setup.html#download-and-setup to setup tensorflow

To run an example:
    cd ~
    git clone https://github.com/kendricktan/openimif.git
    cd openimif
    python setup.py image.png
