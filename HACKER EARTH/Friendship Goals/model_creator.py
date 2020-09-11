import pandas as pd
import numpy as np
import PIL
import cv2 ,pickle
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model




# test_data = pd.read_csv("dataset/test.csv")
# train_data = pd.read_csv("dataset/train.csv")
#
#
# print(test_data.head())
# print(train_data.head())
#












validation_dir="dataset/validation"





train_data=[]

train = []
for i in ["Adults","Teenagers","Toddler"]:
    path = "dataset/data/train/"+i
    for each in os.listdir(path):
        img_data = cv2.imread(path+'/'+each)
        img_data = cv2.resize(img_data, (400, 400),
                               interpolation=cv2.INTER_NEAREST)
        train.append(np.array(img_data))
        train_data.append(i)



test = []
for i in ["test"]:
  path = "dataset/test"
  for each in os.listdir(path):
      img_data = cv2.imread(path+'/'+each)
      img_data = cv2.resize(img_data, (400, 400),
                               interpolation=cv2.INTER_NEAREST)
      test.append(np.array(img_data))


print(train[0].shape)


import pickle
pickle.dump(train,open("dataset/train.npy","wb"))
pickle.dump(test, open("dataset/test.npy","wb"))





test_img = pickle.load(open("dataset/test.npy","rb"))
train_img = pickle.load(open("dataset/train.npy","rb"))



train_img = np.array(train_img)
test_img = np.array(test_img)


# print(train_data.head())
# print(test_data.head())



encoder = LabelEncoder()
encoder.fit(train_data)
encoded_Y = encoder.transform(train_data)
dummy_y = np_utils.to_categorical(encoded_Y)


print(dummy_y.shape)

print(train_img.shape)


from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.wrappers.scikit_learn import KerasClassifier



from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.wrappers.scikit_learn import KerasClassifier




from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD







base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(400, 400, 3),
    pooling='max'
)


base_model.trainable = False

model = Sequential([
  base_model,
    Dense(128,activation='relu'),
  Dense(3,kernel_regularizer=regularizers.l2(0.005), activation='softmax')
])

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger
# checkpoint = ModelCheckpoint('dataset/best_model.h5',
#                              monitor='val_loss',
#                              verbose=1, save_best_only= True,
#                              mode='auto')


checkpointer = ModelCheckpoint(filepath='best_model_8class.hdf5',monitor='accuracy', verbose=1, save_best_only=True)
csv_logger = CSVLogger('history_8class.log')



batch_size=2
epochs = 10


validation_data_dir = validation_dir



test_datagen = ImageDataGenerator()


validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(400, 400),
    batch_size=batch_size,
    class_mode='categorical')



datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

training_generator = datagen.flow(train_img, dummy_y,
                                  batch_size=batch_size)

history = model.fit_generator(
         training_generator,
        validation_data=validation_generator,
        validation_steps=validation_generator.n//validation_generator.batch_size,
         steps_per_epoch= training_generator.n//training_generator.batch_size,
         callbacks=[checkpointer,csv_logger],
         epochs= epochs)

model.save('model_trained_8class.hdf5')


# Visualize the accuracy and loss plots

#model = load_model('best_model_8class.hdf5',compile = False)

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


def graph(history,title):
    plt.title(title)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_accuracy', 'validation_accuracy','train_loss', 'validation_loss'], loc='best')
    plt.show()
    plt.savefig('DANCE_FORM_8-ResNet50.jpg')


plot_accuracy(history,'DANCE_FORM_8-ResNet50')
plot_loss(history,'DANCE_FORM_8-ResNet50')
graph(history,'DANCE_FORM_8-ResNet50')




pred = model.predict(test_img)
print(pred)
pred = np.argmax(pred, axis=1)
test_data = pd.read_csv("dataset/Test.csv")

pred = encoder.inverse_transform(pred)
print(pred)
result = pd.DataFrame(pred, test_data["Filename"], columns=["Category"])
result.to_csv("dataset/sample.csv")