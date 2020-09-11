import pandas as pd
import numpy as np
import PIL
import cv2 ,pickle
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pickle



model = load_model('best_model_8class.hdf5',compile = False)

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


# plot_accuracy(model.history,'DANCE_FORM_8-ResNet50')
# plot_loss(model.history,'DANCE_FORM_8-ResNet50')
# graph(model.history,'DANCE_FORM_8-ResNet50')


test_img = pickle.load(open("dataset/test.npy","rb"))


train_data=[]

train = []
for i in ["Adults","Teenagers","Toddler"]:
    path = "dataset/data/train/"+i
    for each in os.listdir(path):
        img_data = cv2.imread(path+'/'+each)
        img_data = cv2.resize(img_data, (224, 224),
                               interpolation=cv2.INTER_NEAREST)
        train.append(np.array(img_data))
        train_data.append(i)




encoder = LabelEncoder()
encoder.fit(train_data)
encoded_Y = encoder.transform(train_data)
dummy_y = np_utils.to_categorical(encoded_Y)


pred = model.predict(test_img)
print(pred)
pred = np.argmax(pred, axis=1)
test_data = pd.read_csv("dataset/Test.csv")

pred = encoder.inverse_transform(pred)
print(pred)
result = pd.DataFrame(pred, test_data["Filename"], columns=["Category"])
result.to_csv("dataset/sample.csv")