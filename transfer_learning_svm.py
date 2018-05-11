"""
Image Classification - Transfer Learning with SVM

The Data should be organized as 

../data/train/class
../data/validation/class
Here class is indicative of the Categories of Classification. For Example in Dogs vs Cats Classifier the directory could be
../data/train/dogs
"""

import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import math

import sklearn
from sklearn import cross_validation, grid_search
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn import svm
from sklearn.externals import joblib

from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
# dimensions of our images.
img_width, img_height = 224, 224

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

# number of epochs to train top model
epochs = 80
# batch size used by flow_from_directory and predict_generator
batch_size = 16


def save_bottlebeck_features():
    # We could build any of the Deep CNNs like VGG-16 or Inception-V3 here
    model = applications.VGG16(include_top=False, weights='imagenet')        
    #model = applications.InceptionV3(include_top=False, weights='imagenet')
    #model.load_weights('drive/inception_top.h5')
    datagen = ImageDataGenerator(rescale=1. / 255)

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    """
    print(len(generator.filenames))
    print(generator.class_indices)
    print(len(generator.class_indices))
    """
    nb_train_samples = len(generator.filenames)
    num_classes = len(generator.class_indices)

    predict_size_train = int(math.ceil(nb_train_samples / batch_size))

    bottleneck_features_train = model.predict_generator(
        generator, predict_size_train)
    

    # Saving the Bottleneck features 	
    np.save('bottleneck_features_train.npy', bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    nb_validation_samples = len(generator.filenames)

    predict_size_validation = int(
        math.ceil(nb_validation_samples / batch_size))

    bottleneck_features_validation = model.predict_generator(
        generator, predict_size_validation)

    np.save('bottleneck_features_validation.npy',
            bottleneck_features_validation)


def train_top_model():
    datagen_top = ImageDataGenerator(rescale=1. / 255)
    # Here Classmode can be 'binary' for binary classification and 'categorical' for multi-class classification
	generator_top = datagen_top.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary',                                      
        shuffle=False)

    nb_train_samples = len(generator_top.filenames)
    num_classes = len(generator_top.class_indices)

    # save the class indices to use use later in predictions
    np.save('class_indices.npy', generator_top.class_indices)

    # load the bottleneck features saved earlier
    train_data = np.load('bottleneck_features_train.npy')

    # get the class labels for the training data, in the original order
    train_labels = generator_top.classes

 
    # convert the training labels to categorical vectors
    train_labels = to_categorical(train_labels, num_classes=num_classes)

    generator_top = datagen_top.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    nb_validation_samples = len(generator_top.filenames)

    validation_data = np.load('bottleneck_features_validation.npy')

    validation_labels = generator_top.classes
    validation_labels = to_categorical(
        validation_labels, num_classes=num_classes)
    
    dataset_size = nb_train_samples
    TwoDim_dataset = train_data.reshape(dataset_size,-1)
    
    print("Fitting the classifier to the training set")
    t0 = time()
    # Perform Grid Search to Arrive at Best Possible parameters for the SVM Classifier
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced',decision_function_shape='ovo'), param_grid)
    clf.fit(TwoDim_dataset,train_labels)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)

    v_dataset_size=nb_validation_samples
    TwoDim_vdataset = validation_data.reshape(v_dataset_size,-1)
    #y_predict=clf.predict(TwoDim_vdataset)
    
    print("Predicting people's names on the test set")
    t0 = time()
    y_pred = clf.predict(TwoDim_vdataset) 
    print("done in %0.3fs" % (time() - t0))
    # Metrics that are indicative of the performance of the Model
    print(classification_report(validation_labels, y_pred))
    print(confusion_matrix(validation_labels, y_pred, labels=range(3)))
    

save_bottlebeck_features()
train_top_model()