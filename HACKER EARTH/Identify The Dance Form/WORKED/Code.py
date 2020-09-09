import pandas as pd
import numpy as np
import PIL
import cv2 ,pickle
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


test_data = pd.read_csv("dataset/test.csv")
train_data = pd.read_csv("dataset/train.csv")


print(test_data.head())
print(train_data.head())




train = []
for i in train_data["Image"]:
    path = "dataset/train/"+i
    img_data = cv2.imread(path)
    img_data = cv2.resize(img_data, (224, 224),
                           interpolation=cv2.INTER_NEAREST)
    train.append(np.array(img_data))


test = []
for i in test_data["Image"]:
  path = "dataset/test/"+i
  img_data = cv2.imread(path)
  img_data = cv2.resize(img_data, (224, 224),
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


print(train_data.head())
print(test_data.head())



encoder = LabelEncoder()
encoder.fit(train_data["target"])
encoded_Y = encoder.transform(train_data["target"])
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


base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3),
    pooling='max'
)


base_model.trainable = False

model = Sequential([
  base_model,
  Dropout(0.3),
  Dense(8, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger
# checkpoint = ModelCheckpoint('dataset/best_model.h5',
#                              monitor='val_loss',
#                              verbose=1, save_best_only= True,
#                              mode='auto')


checkpointer = ModelCheckpoint(filepath='best_model_8class.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('history_8class.log')



batch_size =5
epochs = 100


datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

training_generator = datagen.flow(train_img, dummy_y,
                                  batch_size=batch_size)

history = model.fit_generator(
         training_generator,
         steps_per_epoch= training_generator.n//training_generator.batch_size,
         callbacks=[checkpointer,csv_logger],
         epochs= epochs)

model.save('model_trained_8class.hdf5')




pred = model.predict(test_img)
print(pred)
pred = np.argmax(pred, axis=1)
test_data = pd.read_csv("dataset/test.csv")

pred = encoder.inverse_transform(pred)
print(pred)
result = pd.DataFrame(pred, test_data["Image"], columns=["target"])
result.to_csv("dataset/sample.csv")




