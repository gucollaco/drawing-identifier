# drawing-identifier

A university project to apply a multilayer perceptron network onto an algorithm, so that it can learn to identify the classes of some images (used on the training process). When you provide an image for it to judge, you will be told the probability of this image to be each of the trained classes.

Firstly, check if you have the following libraries installed (and if possible, update them):
- Keras
- NumPy
- Scikit-learn
- Matplotlib
- PIL

Then, select on [Quick, Draw!](https://quickdraw.withgoogle.com/data), some .npy data to be used on the experiment. Once you selected the desired datas, put them inside the 'data' folder (the code will use these files on the learning process).

The next step is to execute the main.py file, which will accomplish the training process, and store a keras model at the 'model' folder.
```
python main.py
```
Once this is done, run the verify.py file, which will check if the image you want to classify (you can use one of the images on the 'my_drawings' folder), and plot a chart containing the information seeked.
```
python verify.py
```
