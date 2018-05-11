"""
Image Classification - Training a small CNN from Scratch

The Data should be organized as 

../data/train/class
../data/validation/class
Here class is indicative of the Categories of Classification. For Example in Dogs vs Cats Classifier the directory could be
../data/train/dogs
"""

from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
import os
#%%

# dimensions of our images.
img_width, img_height = 299, 299

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2525
nb_validation_samples = 631
epochs = 20
batch_size = 16
#%%
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
#%%
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.8))
model.add(Dense(1))
model.add(Activation('sigmoid'))
#%%
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
#%%
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
#%%
# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')
#%%
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')
#%%
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)
#%%
model.save_weights('first_try.h5')
model.save('CNN_from_scratch.h5')
#%%


img_width, img_height = 299, 299
model_path = 'first_try.h5'
model_weights_path = 'first_try.h5'
#model = load_model(model_path)
model.load_weights(model_weights_path)

def predict(file):
  x = load_img(file, target_size=(img_width,img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = model.predict(x)
  
  result = array[0]
  if result==0:
    print("Predicted answer: cancer")
    answer = 'cancer'
  else:
    print("Predicted answer: normal")
    answer = 'normal'

  return answer

tp = 0
tn = 0
fp = 0
fn = 0

for i, ret in enumerate(os.walk('data/validation/cancer')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    print("Label: cancer")
    result = predict(ret[0] + '/' + filename)
    if result == "cancer":
      tn += 1
    else:
      fp += 1

for i, ret in enumerate(os.walk('data/validation/normal')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    print("Label: normal")
    result = predict(ret[0] + '/' + filename)
    if result == "normal":
      tp += 1
    else:
      fn += 1

"""
Check metrics
"""
print("True Positive: ", tp)
print("True Negative: ", tn)
print("False Positive: ", fp)  # important
print("False Negative: ", fn)

precision = tp / (tp + fp)
recall = tp / (tp + fn)
specificity =(tn)/(tn+fp)
sensitivity =(tp)/(tp+fn)
print("Precision: ", precision)
print("Recall: ", recall)
print("Specificity:",specificity)
print("Sensitivity:",sensitivity)
f_measure = (2 * recall * precision) / (recall + precision)
print("F-measure: ", f_measure)
