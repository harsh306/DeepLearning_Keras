# Baseline MLP for MNIST dataset
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.optimizers import *
from keras.layers import Dropout
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
# define baseline model
print(num_pixels,num_classes)
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, init='normal', activation='relu'))#input_shape=((1,28,28,))
    model.add(Dropout(0.4))
    model.add(Dense(num_pixels*4, init='normal', activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_pixels//2, init='normal', activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_pixels//4, init='normal', activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_pixels//4, init='normal', activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, init='normal', activation='softmax'))
    # Compile model
    #optimizers
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) #1
    sgd = SGD(lr=0.001, momentum=0.5, decay=0.0, nesterov=False)#5
    adgrad = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)#2
    adadelta = Adadelta(lr=10.0, rho=0.95, epsilon=1e-08, decay=0.0)#1
    rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0) #1
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model
# build the model
model = baseline_model()
# Fit the model
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=20, batch_size=200, verbose=2,callbacks=[learning_rate_reduction])
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.4f%%" % (100-scores[1]*100))
