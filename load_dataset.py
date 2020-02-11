# import required packages and libraries.

import os
import cv2
import urllib
import csv
import time
# import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from pylab import rcParams
from tqdm import tqdm
from matplotlib import rc
# from pandas.plotting import register_matplotlib_converters
from sklearn.model_selection import train_test_split
from PIL import Image

# path tp the images dataset.
dataset_path = "./image_dataset/"


# Labels used for the daataset accordingly.
label_names = ["Dog", "Cat", "Car"]

# read the image and display
for label in label_names:
    images_path = os.path.join(dataset_path, label)
    for image in os.listdir(images_path):
        sample_image = cv2.imread(os.path.join(
            images_path, image), cv2.IMREAD_GRAYSCALE)
        plt.imshow(sample_image, cmap='gray')
        plt.show()
        break
    break

print(sample_image.shape)

image_size = 70
resized_image = cv2.resize(sample_image, (image_size, image_size))
plt.imshow(resized_image, cmap='gray')
plt.show()

print(resized_image.shape)