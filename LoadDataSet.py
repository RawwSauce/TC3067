# -*- coding: utf-8 -*-
"""
Created on Mon May 25 21:35:27 2020

@author: renet
"""
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob

# initialize image lists
alien_images_list = []
predator_images_list = []

#load alien images
for filename in glob.glob('alien-vs-predator-images/data/alien/*.jpg'):
    im=plt.imread(filename)
    alien_images_list.append(im)
    
#convert list to array
alien_images_array = np.stack(alien_images_list,axis=0)
alien_class_array = np.ones((1,347))

#load alien images
for filename in glob.glob('alien-vs-predator-images/data/predator/*.jpg'):
    im=plt.imread(filename)
    predator_images_list.append(im)
    
#convert list to array
predator_images_array = np.stack(predator_images_list,axis=0)
predator_class_array = np.zeros((1,347))

print(alien_images_array.shape)
print(predator_images_array.shape)

#concatenate two data sets and create class vector
image_dataset = np.concatenate((alien_images_array,predator_images_array),axis=0)
class_dataset = np.concatenate((alien_class_array,predator_class_array),axis=1)
print(image_dataset.shape)
print(class_dataset.shape)

randomize = np.arange(len(image_dataset))
np.random.shuffle(randomize)
image_dataset = image_dataset[randomize]
class_dataset = (class_dataset.transpose())[randomize].transpose()

print(image_dataset.shape)
print(class_dataset.shape)

# Example of a picture
index = 693
plt.imshow(image_dataset[index])
if int(class_dataset[0,index]) == 1:
    print ("y = " + str(int(class_dataset[0,index])) + ". It's an alien picture.")
else:
    print ("y = " + str(int(class_dataset[0,index])) + ". It's a predator picture.")
    
train_x_orig = np.concatenate((alien_images_array[:250],predator_images_array[:250]),axis=0)
train_y = np.array([np.concatenate((alien_class_array[0,:250],predator_class_array[0,:250]),axis=0)])
dev_x_orig = np.concatenate((alien_images_array[250:297],predator_images_array[250:297]),axis=0)
dev_y = np.concatenate((alien_class_array[0,250:297],predator_class_array[0,250:297]),axis=0)
test_x_orig = np.concatenate((alien_images_array[297:],predator_images_array[297:]),axis=0)
test_y = np.concatenate((alien_class_array[0,297:],predator_class_array[0,297:]),axis=0)

print ("Number of training examples: " + str(train_x_orig.shape[0]))
print ("Number of dev examples: " + str(dev_x_orig.shape[0]))
print ("Number of testing examples: " + str(test_x_orig.shape[0]))
print ("Each image is of size: (" + str(train_x_orig.shape[1]) + ", " + str(train_x_orig.shape[2]) + ", " + str(train_x_orig.shape[3])+ ")")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("dev_x_org shape: " + str(dev_x_orig.shape))
print ("dev_y shape: " + str(dev_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))