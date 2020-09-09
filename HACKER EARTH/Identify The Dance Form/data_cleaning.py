import os
import pandas as pd
import shutil, os
import random


# Setting up Location For the Directory

dir_path=os.getcwd()
dir_path=dir_path.replace("\\","/")

print(dir_path)




# The path to the dataset Directory
base_directory=dir_path+'/dataset'

# The path to the training images directory
train_dir=base_directory+'/train'

# The path to the test images directory
test_dir=base_directory+'/test'


train_csv_data=pd.read_csv(base_directory+'/train.csv')

print(train_csv_data.head())




dance_class_types=train_csv_data.target.unique().tolist()


print(dance_class_types)





dir_train='DATA/train'
dir_test='DATA/test'


if not os.path.exists('DATA'):
    os.mkdir('DATA')
else:
    print("Directory " , 'DATA' ,  " already exists")


# Create target Training Directory if don't exist
if not os.path.exists(dir_train):
    os.mkdir(dir_train)
    os.mkdir(dir_test)
    print("Directory " , dir_train ,  " Created ")
    print("Directory ", dir_test, " Created ")
    for each in dance_class_types:
        if not os.path.exists(dir_train+'/'+each):
            os.mkdir(dir_train+'/'+each)
            print("Sub Directory ", dir_train+'/'+each, " Created ")
        else:
            print("Sub Directory ", dir_train + '/' + each, " already exists ")

    for each in dance_class_types:
        if not os.path.exists(dir_test+'/'+each):
            os.mkdir(dir_test+'/'+each)
            print("Sub Directory ", dir_test+'/'+each, " Created ")
        else:
            print("Sub Directory ", dir_test + '/' + each, " already exists ")

else:
    print("Directory " , dir_train ,  " already exists")
    print("Directory ", dir_test, " already exists")




print(dir_path)
print(train_dir)

training_dir=dir_path+'/'+dir_train
testing_dir=dir_path+'/'+dir_test


directory_structure=train_csv_data.values.tolist()



print(directory_structure)



# Copying the images from the dataset/train to their individual subfolders according to their image classes  in training folders
for f in directory_structure:
    toss=random.randint(0,10)
    if toss<=8:
        shutil.copy(train_dir+'/'+f[0], training_dir+'/'+f[1])
    else:
        shutil.copy(train_dir + '/' + f[0], testing_dir + '/' + f[1])

