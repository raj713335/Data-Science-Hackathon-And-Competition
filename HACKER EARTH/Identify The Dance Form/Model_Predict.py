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
import pandas as pd


import warnings
warnings.filterwarnings('ignore')


# Checking if GPU is enabled

# Check if GPU is enabled
print(tf.__version__)
print(tf.test.gpu_device_name())


# Setting up Location For the Directorys

dir_path=os.getcwd()
dir_path=dir_path.replace("\\","/")

# The path to the dataset Directory
base_directory=dir_path+'/dataset'

# The path to the training images directory
train_dir=base_directory+'/train'

# The path to the test images directory
test_dir=base_directory+'/test'

# Creating a list variable containing the names of all the images
test_csv_data=pd.read_csv(base_directory+'/test.csv')

predict_images_list=test_csv_data.Image.values.tolist()



train_csv_data=pd.read_csv(base_directory+'/train.csv')
dance_class_types=train_csv_data.target.unique().tolist()

print(predict_images_list)
print(len(predict_images_list))




# Loading the created model from running the Model_Creator.py file.

# Loading the best saved model to make predictions
model = load_model('best_model_8class.hdf5',compile = False)

# Loading The Model Summary
model.summary()


# Predicting the values of test images

# for each in predict_images_list:
#     predict_class(model_best, test_dir+'/'+each, True)




# Defining Helper Functions

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def generate_pattern(layer_name, filter_index, size=150):
    # Build a loss function that maximizes the activation
    # of the nth filter of the layer considered.
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # Compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, model.input)[0]

    # Normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # This function returns the loss and grads given the input picture
    iterate = K.function([model.input], [loss, grads])

    # We start from a gray image with some noise
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.

    # Run gradient ascent for 40 steps
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    return deprocess_image(img)




def get_activations(img, model_activations):
    img = image.load_img(img, target_size=(299, 299))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.
    plt.imshow(img[0])
    plt.show()
    return model_activations.predict(img)


def show_activations(activations, layer_names):
    images_per_row = 16

    # Now let's display our feature maps
    for layer_name, layer_activation in zip(layer_names, activations):
        # This is the number of features in the feature map
        n_features = layer_activation.shape[-1]

        # The feature map has shape (1, size, size, n_features)
        size = layer_activation.shape[1]

        # We will tile the activation channels in this matrix
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        # We'll tile each filter into this big horizontal grid
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                :, :,
                                col * images_per_row + row]
                # Post-process the feature to make it visually palatable
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size,
                row * size: (row + 1) * size] = channel_image

        # Display the grid
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')

    plt.show()


print(len(model.layers))


# We start with index 1 instead of 0, as input layer is at index 0
layers = [layer.output for layer in model.layers[1:11]]
# We now initialize a model which takes an input and outputs the above chosen layers
activations_output = models.Model(inputs=model.input, outputs=layers)


print(layers)


#Get the names of all the selected layers

layer_names = []
for layer in model.layers[1:11]:
    layer_names.append(layer.name)
print(layer_names)


# Visualize the activations of intermediate layers from layer 1 to 10

for i in range(0,1):
    toss=random.randint(0,len(predict_images_list)-1)
    activations = get_activations(test_dir+'/'+predict_images_list[toss] ,activations_output)
    show_activations(activations, layer_names)


# Get the index of activation_1 layer which has sparse activations
ind = layer_names.index('activation_1')
sparse_activation = activations[ind]
a = sparse_activation[0, :, :, 13]
print(a)

print(all (np.isnan(a[j][k])  for j in range(a.shape[0]) for k in range(a.shape[1])))

# Get the index of batch_normalization_1 layer which has sparse activations
ind = layer_names.index('batch_normalization_1')
sparse_activation = activations[ind]
b = sparse_activation[0, :, :, 13]
print(b)


#Show the activation outputs of 1st, 2nd and 3rd Conv2D layer activations to compare how layers get abstract with depth

first_convlayer_activation = activations[0]
second_convlayer_activation = activations[3]
third_convlayer_activation = activations[6]
f,ax = plt.subplots(1,3, figsize=(10,10))
ax[0].imshow(first_convlayer_activation[0, :, :, 3], cmap='viridis')
ax[0].axis('OFF')
ax[0].set_title('Conv2d_1')
ax[1].imshow(second_convlayer_activation[0, :, :, 3], cmap='viridis')
ax[1].axis('OFF')
ax[1].set_title('Conv2d_2')
ax[2].imshow(third_convlayer_activation[0, :, :, 3], cmap='viridis')
ax[2].axis('OFF')
ax[2].set_title('Conv2d_3')


def get_attribution(food):
    img = image.load_img(food, target_size=(299, 299))
    img = image.img_to_array(img)
    img /= 255.
    f, ax = plt.subplots(1, 3, figsize=(15, 15))
    ax[0].imshow(img)

    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    class_id = np.argmax(preds[0])
    ax[0].set_title("Input Image")
    class_output = model.output[:, class_id]
    last_conv_layer = model.get_layer("mixed10")

    # grads = K.gradients(class_output, last_conv_layer.output)[0]
    # pooled_grads = K.mean(grads, axis=(0, 1, 2))
    # iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    # pooled_grads_value, conv_layer_output_value = iterate([img])
    # for i in range(2048):
    #     conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    #
    # heatmap = np.mean(conv_layer_output_value, axis=-1)
    # heatmap = np.maximum(heatmap, 0)
    # heatmap /= np.max(heatmap)
    # ax[1].imshow(heatmap)
    # ax[1].set_title("Heat map")

    # act_img = cv2.imread(food)
    # heatmap = cv2.resize(heatmap, (act_img.shape[1], act_img.shape[0]))
    # heatmap = np.uint8(255 * heatmap)
    # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # superimposed = cv2.addWeighted(act_img, 0.6, heatmap, 0.4, 0)
    # cv2.imwrite('classactivation.png', superimposed)
    # img_act = image.load_img('classactivation.png', target_size=(299, 299))
    # ax[2].imshow(img_act)
    # ax[2].set_title("Class Activation")
    # plt.show()
    return preds



#Getting Heat Maps in Image
for i in range(0,2):
    toss=random.randint(0,len(predict_images_list)-1)
    pred = get_attribution(test_dir+'/'+predict_images_list[toss])

    print(pred)


output_list=[]

for i in range(0,len(predict_images_list)):
    pred = get_attribution(test_dir+'/'+predict_images_list[i]).tolist()
    pred=pred[0]
    dance_form_id=pred.index(max(pred))
    print(pred.index(max(pred)))
    output_list.append([predict_images_list[i],dance_class_types[dance_form_id]])


dfObj = pd.DataFrame(output_list, columns = ['Image','target'])


dfObj.to_csv('test.csv',index=False)



