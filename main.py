# -*- coding: utf-8 -*-

# required imports
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras import optimizers
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import os

# read the .npy files, collecting them as np arrays
def load_files(data_path, files):
    data = []
    for file in files:
        f = np.load(data_path + file)
        data.append(f)
    return data

# normalize the values
def normalize(values):
    return (values - values.min()) / (values.max() - values.min())

# display a read image, after loading it, if you want to see if it is reading properly
def show_image(array):
    array = np.reshape(array, (28, 28))
    img = Image.fromarray(array)
    return img.show()

# for each np array inside the arrays, we are going to limit the quantity of samples
def limit_qty(arrays, n):
    new_data = []
    for array in arrays:
        i = 0
        for element in array:
            if i == n: break
            new_data.append(normalize(element))
            i += 1
    return new_data

# according to the quantity of samples, we are generating the labels (at this moment, they are ordered)
def define_labels(n_files, n_samples):
    labels = []
    for i in range(n_files):
        labels += [i] * n_samples
    return labels

# function to execute the process
def execute():
    # path to access the .npy files
    data_path = os.getcwd() + '/data/'
    
    # list of files, and quantity of files inside the list
    files = os.listdir(data_path)
    n_files = len(files)
    
    # some constants definition
    n_samples = 1000
    n_epochs = 20
    
    # returning the read data, and limiting the quantity according to the n_samples constant
    objects = load_files(data_path, files)
    objects = limit_qty(objects, n_samples)
    
    # defining the labels
    labels = define_labels(n_files, n_samples)
    
    # separating the sets and labels between training and test (0.9 and 0.1, proportion)
    training_set, test_set, training_labels, test_labels = train_test_split(objects, labels, test_size=0.1)
    
    # transforming index 2, into [0, 0, 1], for example
    training_labels_modified = np_utils.to_categorical(training_labels, n_files)
  
    # adding layers to the MLP
    model = Sequential()
    model.add(Dense(units=500, activation='relu', input_dim=784))
    model.add(Dropout(0.3))
    model.add(Dense(units=500, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units=250, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units=25, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units=n_files, activation='softmax'))
    
    # stochastic gradient descent
    sgd = optimizers.SGD(lr=0.1, momentum=0.8)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    
    # fit as batches
    model.fit(np.array(training_set), np.array(training_labels_modified), batch_size=32, epochs=n_epochs)
    print('______________________')
    print('Finished training')
    
    # predict test_set
    predictions = model.predict(np.array(test_set))
    
    # checks how many were 'on the target'
    score = 0
    for i in range(len(predictions)):
        if np.argmax(predictions[i]) == test_labels[i]: score += 1
    print('______________________')
    print('Accuracy: ', ((score + 0.0) / len(predictions)) * 100)
    
    # saving the keras model to be applied on the other script
    model.save("model/drawings.h5")
    print('______________________')
    print('Model generated and saved')
    
# execute function call, on main (not executing on import)
if __name__ == "__main__":
   execute()