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
classifier.add(Conv2D(32,(3,3), input_shape = (48,48,1), activation = 'relu'))

#step 2: Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

#add a second Convolutional layer
classifier.add(Conv2D(32,(3,3), activation = 'relu'))
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
        target_size=(48, 48),
        color_mode = 'grayscale',
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(48, 48),
        color_mode = 'grayscale',
        batch_size=32,
        class_mode='binary')

#Train the Model
from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]



history =classifier.fit_generator(
        generator = train_set,
        #number of images in our train_set
        steps_per_epoch=10,
        epochs=25,
        validation_data=test_set,
        #number of images in our test_set
        validation_steps=2,
        callbacks = callbacks_list
        )

# test_eval = classifier.predict(test_set, verbose=0)
# predict = classifier.predict('dataset/test_set', batch_size = 32)
# print(test_eval)
# print(predict)
import numpy as np
from keras.preprocessing import image
# test_image = image.load_img('pigfoot2.jpeg',target_size =  (64,64))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image,axis=0)
# result = classifier.predict(test_image)
# train_set.class_indices
# if result[0][0]>=0.5:
#     prediction = 'healthy'
# else:
#     prediction = 'sick'
# print(prediction)
# print(result[0][0])





# serialize model structure to JSON
model_json = classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)


# plot the evolution of Loss and Acuracy on the train and validation sets

import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
plt.suptitle('Optimizer : Adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.show()
