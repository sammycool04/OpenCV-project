from keras.models import Sequential
# from keras.layers import Convolution2D
from keras.layers import Conv2D

from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Initialising CNN
classifier = Sequential()


#step 1: Convolution
classifier.add(Conv2D(32,3,3, input_shape = (64,64,3), activation = 'relu'))

#step 2: Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

#add a second Convolutional layer
classifier.add(Conv2D(32,3,3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))


#step 3: Flattening
classifier.add(Flatten())

#step 4: Full Connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

#step 5: compile CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fit the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
        'dataset/train_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        train_set,
        #number of images in our train_set
        steps_per_epoch=10,
        epochs=25,
        validation_data=test_set,
        #number of images in our test_set
        validation_steps=2)

test_eval = classifier.predict(test_set, verbose=0)
predict = classifier.predict('dataset/test_set', batch_size = 32)
print(test_eval)
print(predict)
