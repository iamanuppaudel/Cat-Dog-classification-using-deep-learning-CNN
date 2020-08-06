#%%
import tensorflow as tf
#import keras
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D 
import pickle
from tensorflow.keras.callbacks import TensorBoard
import time
import numpy as np

NAME= "Cats_Dogs_classifier_CNN_{}".format(int(time.time()))

X=pickle.load(open("X.pickle","rb"))
y=pickle.load(open("y.pickle","rb"))
y=np.array(y)
X= X/255.0

model =Sequential()

model.add(Conv2D(64,kernel_size=(3,3), activation='relu', input_shape=X.shape[1:]))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(80,kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(90, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

tensorboard= TensorBoard(log_dir="logs\{}".format(NAME))

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])



model.fit(X,y, batch_size=50, epochs=5, validation_split=0.2, callbacks=[tensorboard])


# %%
