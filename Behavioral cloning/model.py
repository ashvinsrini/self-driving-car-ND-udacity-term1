import pandas as pd
import numpy as np
import cv2
import os
import glob
import pdb
import math
from keras.layers import Cropping2D
from scipy import ndimage
from keras.layers import Lambda
from keras.models import Sequential
from keras.layers import Cropping2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import csv
import pdb
samples = []
os.chdir('/home/workspace')
drivinglogPath = os.path.join(os.getcwd(),'CarND-Behavioral-Cloning-P3','data','driving_log.csv')
##### Following code is to read the csv file of the log data that contains the images filepath and the target variable
with open(drivinglogPath) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
###### Creating train and validation samples. ############
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import numpy as np
import sklearn
#path = os.path.join(os.getcwd(),'CarND-Behavioral-Cloning-P3','data')
##### Creating generators that reads a huge data on the fly  ######

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        #shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            #### Create a series of lists to store the images 
            centerimages = []
            leftimages = []
            rightimages = []
            centerangles = []
            leftangles = []
            rightangles = []
            for batch_sample in batch_samples:
                if batch_sample[3]!= 'steering':
                    path = os.path.join(os.getcwd(),'CarND-Behavioral-Cloning-P3','data','IMG')
                    #pdb.set_trace()
                    centername = path+'/'+batch_sample[0].split('/')[-1]
                    center_image = cv2.imread(centername)
                    center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)   # Convert BGR to RGB format 
                    leftname = path+'/'+batch_sample[1].split('/')[-1]
                    left_image = cv2.imread(leftname)
                    left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                    rightname = path+'/'+batch_sample[2].split('/')[-1]
                    right_image = cv2.imread(rightname)
                    right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                    
                    #print('angle is {}'.format(batch_sample[3]))
                    #### Account for distortion correction######
                    center_angle = float(batch_sample[3])
                    left_angle = float(batch_sample[3]) + 0.2 #### Add a distortion factor of 0.2 degrees for left images.
                    right_angle = float(batch_sample[3]) - 0.2 #### Subtract a distorition factor of 0.2 degrees for right images
                    
                    centerimages.append(center_image)
                    leftimages.append(left_image)
                    rightimages.append(right_image)
                    
                    centerangles.append(center_angle)
                    leftangles.append(left_angle)
                    rightangles.append(right_angle)
                    

            # trim image to only see section with road
            #X_train = np.array(images)
            #y_train = np.array(angles)
            X_train = np.concatenate((np.array(centerimages),np.array(leftimages),np.array(rightimages)))

            y_train = np.concatenate((np.array(centerangles),np.array(leftangles),np.array(rightangles)))
            #pdb.set_trace()
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

row, col, ch =  90, 320, 3  # Trimmed image format
###### Declaring model architecture. ##########
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3))) ####### Cropping image the top 50 pixels and bottom 20 pixels
model.add(Lambda(lambda x: x/255 - 0.5,input_shape=(row, col, ch),output_shape=(row, col, ch))) ### Feature normalisation. 
model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))   ##### Using  convolution layer of depth 24 with 5 by 5 filter size 
model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Dropout(0.75))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))
print('printing model summary -------\n')
print(model.summary())
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=math.ceil(len(validation_samples)/batch_size), epochs=10, verbose=1)

os.chdir('ProjectFinalSubmission')
model.save('model.h5')
