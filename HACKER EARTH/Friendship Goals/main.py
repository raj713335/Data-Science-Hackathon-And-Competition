import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk(''):
    for filename in filenames:
        print(os.path.join(dirname, filename))



labels = os.listdir('dataset/train')
print(labels)


num = []
for label in labels:
    path = 'train/{0}/'.format(label)
    folder_data = os.listdir(path)
    k = 0
    print('\n', label.upper())
    for image_path in folder_data:
        #if k < 5:
            #display(Image(path+image_path))
        k = k+1
    num.append(k)
    print('there are ', k,' images in ', label, 'class')



import matplotlib.pyplot as plt
plt.figure(figsize = (8,8))
plt.bar(labels, num)
plt.title('NUMBER OF IMAGES CONTAINED IN EACH CLASS')
plt.xlabel('classes')
plt.ylabel('count')
plt.show()



x_data =[]
y_data = []
import cv2
for label in labels:
    path = 'train/{0}/'.format(label)
    folder_data = os.listdir(path)
    for image_path in folder_data:
        image = cv2.imread(path+image_path)
        #image_resized = cv2.resize(image, None,fx=2, fy=2,interpolation = cv2.INTER_NEAREST)
        x_data.append(np.array(image))
        y_data.append(label)





x_data = np.array(x_data)
y_data = np.array(y_data)
print('the shape of X is: ', x_data.shape, 'and that of Y is: ', y_data.shape)




#stadardizing the input data
x_data = x_data/255



#converting the y_data into categorical:
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
ohe = OneHotEncoder(sparse=False)
y_data = le.fit_transform(y_data)
y_data = ohe.fit_transform(y_data.reshape(-1,1))



inv_y_data = ohe.inverse_transform(y_data)
inv_y_data = le.inverse_transform(inv_y_data.astype(int).ravel())

print(inv_y_data)



X_train = x_data
Y_train = y_data


xt_data = []
path = 'dataset/test/'
folder_data = os.listdir(path)
for image_path in folder_data:
    image = cv2.imread(path+image_path)
    image_resized = cv2.resize(image, (128,128))
    xt_data.append(np.array(image_resized))




xt_data = np.array(xt_data)
X_test = xt_data.astype('float32')/255



from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(include_top=False,weights='imagenet',input_shape=(150,150,3))
model.summary()



from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D,GlobalAveragePooling2D, AveragePooling2D

x = model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128,activation='relu')(x)
x = Dropout(0.2)(x)

model.summary()



from tensorflow.keras import regularizers

predictions = Dense(3,kernel_regularizer=regularizers.l2(0.005), activation='softmax')(x)

model = Model(inputs=model.input, outputs=predictions)





model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=25, validation_split=0.2)



model.save('model_trained_3class.hdf5')



#Display of the accuracy and the loss values
import matplotlib.pyplot as plt

plt.figure(figsize = (8,8))
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss/accuracy')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()



from keras.utils import plot_model
plot_model(model)



labels = model.predict(X_test)
label = [np.argmax(i) for i in labels]




from sklearn.preprocessing import LabelEncoder
le =LabelEncoder()
target=le.fit(inv_y_data).inverse_transform(label)



test_csv_data=pd.read_csv('C:/Users/Jayita/Downloads/DL on Friendship/data/Test.csv')

submission = pd.DataFrame({ 'Filename': test_csv_data.Filename, 'Category': target })
submission.to_csv('C:/Users/Jayita/Downloads/DL on Friendship/new_out.csv', index=False)





