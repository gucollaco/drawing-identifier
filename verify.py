# -*- coding: utf-8 -*-

# required imports
from main import normalize
from keras.models import load_model
from scipy.misc import imread, imresize
import numpy as np
import os
import matplotlib.pyplot as plt

# data path and files listing
data_path = os.getcwd() + '/data/'
files = os.listdir(data_path)
n_files = len(files)

# removing the extension from the file name and storing on another array
classes = []
for i in range(n_files): classes.append(files[i][:-4])

# turning the files into a dictionary, with the index as its key
expected = dict(enumerate(files))

# loading the saved model
model = load_model('model/drawings.h5')

# reading a specific image, to check which category it is most likely to fit into
x = imread('my_drawings/watermelon_1.png', mode='L')
x = imresize(x, (28, 28))
x = np.invert(x)
x = x.flatten()
x = normalize(x)

# checking the corresponding value
value = model.predict(np.array([x]))
prediction = expected[np.argmax(value)]

# plotting the values
for i in range(n_files): classes[i] += ' (' + str(round(value[0][i], 5)) + '%)'
y_pos = np.arange(len(classes))
plt.bar(y_pos, list(value[0]), align='center', alpha=0.5)
plt.xticks(y_pos, classes)
plt.ylabel('Percentage of being')
plt.title('Drawing identifier')
plt.show()

# printing some info
print('It probably is a: ', prediction)
print('Possibility of being each class: ', list(value[0]))