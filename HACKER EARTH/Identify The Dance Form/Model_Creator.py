# Load the required Libraries


import tensorflow as tf
import matplotlib.image as img
import numpy as np
from collections import defaultdict
import collections
from shutil import copy
from shutil import copytree, rmtree
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os
import random
from tensorflow.keras import regularizers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow import keras
from tensorflow.keras import models
import cv2
import matplotlib.image as mpimg
import PIL


import warnings
warnings.filterwarnings('ignore')


# Checking if GPU is enabled

# Check if GPU is enabled
print(tf.__version__)
print(tf.test.gpu_device_name())


# Setting up Location For the Directorys

dir_path=os.getcwd()
dir_path=dir_path.replace("\\","/")

print(dir_path)

# The path to the dataset Directory
base_directory=dir_path+'/dataset'

# The path to the training images directory
train_dir=base_directory+'/train'

# The path to the test images directory
test_dir=base_directory+'/test'


# Counting total number of images in the test and train directory

print('total training images:', len(os.listdir(train_dir)))
print('total test images:', len(os.listdir(test_dir)))

dir_train='DATA/train'
dir_test='DATA/test'


# Creating a variable containing the directory of the training images classified into sub directoery accoring to their dance forms

training_dir=dir_path+'/'+dir_train
validation_dir=dir_path+'/'+dir_test


print(os.listdir(training_dir))
subdirs = [x[0] for x in os.walk(training_dir)]

print(subdirs)
print(len(subdirs))


print(os.listdir(validation_dir))
subdirs = [x[0] for x in os.walk(validation_dir)]

print(subdirs)
print(len(subdirs))

number_of_train_images=0
number_of_validation_images=0

# Visualize the data, showing one image per class from 101 classes
iter=1
for each in os.listdir(training_dir):
    dance_files = os.listdir(training_dir+'/'+each)
    print(str(iter)+'. '+each.capitalize()+' Images')
    print(dance_files[:])
    number_of_train_images+=len(dance_files[:])
    img_path=training_dir+'/'+each+'/'+str(dance_files[random.randint(0,len(dance_files)-1)])
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.axis('Off')
    plt.title(each)
    plt.xlabel('TRAIN')
    plt.show()
    iter+=1

print(number_of_train_images)





iter=1
for each in os.listdir(validation_dir):
    dance_files = os.listdir(validation_dir+'/'+each)
    print(str(iter)+'. '+each.capitalize()+' Images')
    print(dance_files[:])
    number_of_validation_images+=len(dance_files[:])
    img_path=validation_dir+'/'+each+'/'+str(dance_files[random.randint(0,len(dance_files)-1)])
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.axis('Off')
    plt.title(each)
    plt.xlabel('TEST')
    plt.show()
    iter+=1

print(number_of_train_images)
print(number_of_validation_images)
print(number_of_validation_images+number_of_train_images)





# Training and creating the models

#K.clear_session()
n_classes = 8
img_width, img_height = 150,150
train_data_dir = training_dir
validation_data_dir = validation_dir
nb_train_samples = number_of_train_images
nb_validation_samples = number_of_validation_images
batch_size = 8




train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')



inception = InceptionV3(weights='imagenet', include_top=False)
x = inception.output
x = GlobalAveragePooling2D()(x)
x = Dense(128,activation='relu')(x)
x = Dropout(0.2)(x)

predictions = Dense(8,kernel_regularizer=regularizers.l2(0.005), activation='softmax')(x)

model = Model(inputs=inception.input, outputs=predictions)
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='best_model_8class.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('history_8class.log')

history = model.fit_generator(train_generator,
                    steps_per_epoch = nb_train_samples // batch_size,
                    validation_data=validation_generator,
                    validation_steps=nb_validation_samples // batch_size,
                    epochs=30,
                    verbose=1,
                    callbacks=[csv_logger, checkpointer])


model.save('model_trained_8class.hdf5')


class_map_8 = train_generator.class_indices
print(class_map_8)


# Visualize the accuracy and loss plots

def plot_accuracy(history,title):
    plt.title(title)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_accuracy', 'validation_accuracy'], loc='best')
    plt.show()
def plot_loss(history,title):
    plt.title(title)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'validation_loss'], loc='best')
    plt.show()


plot_accuracy(history,'DANCE_FORM_8-Inceptionv3')
plot_loss(history,'DANCE_FORM_8-Inceptionv3')
