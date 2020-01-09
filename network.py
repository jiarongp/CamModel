import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.activations import softmax
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback

from tensorflow.keras.layers import Input, Dense, Conv2D, Conv3D, Reshape, \
                                    BatchNormalization, Activation, MaxPool2D, Flatten, Dropout

# The callback to be applied at the end of each iteration. This is 
# used to constrain the layer's weights the same way Bayar and Stamm do
# at their paper.
class ConstrainLayer(Model, Callback):
    def __init__(self, model):
        super(ConstrainLayer, self).__init__()
        self.model = model
        self.tmp = None
    # Utilized before each batch
    def on_batch_begin(self, batch, logs={}):
        # Get the weights of the first layer
        all_weights = self.model.get_weights()
        weights = np.asarray(all_weights[0])
        # check if it is converged
        if self.tmp is None or self.tmp != weights:
            # Constrain the first layer
            weights = constrainLayer(weights)
            self.tmp = weights
            # Return the constrained weights back to the network
            all_weights[0] = weights
        self.model.set_weights(all_weights)

def constrainLayer(weights):
    
    # Scale by 10k to avoid numerical issues while normalizing
    weights = weights*10000
    
    # Kernel size is 5 x 5 
    # Set central values to zero to exlude them from the normalization step
    weights[2,2,:,:]=0

    # Pass the weights 
    filter_1 = weights[:,:,0,0]
    filter_2 = weights[:,:,0,1]
    filter_3 = weights[:,:,0,2]
    
    # Normalize the weights for each filter. 
    # Sum in the 3rd dimension, which contains 25 numbers.
    filter_1 = filter_1 / np.sum(filter_1)
    filter_1[2, 2] = -1

    filter_2 = filter_2 / np.sum(filter_2)
    filter_2[2, 2] = -1

    filter_3 = filter_3 / np.sum(filter_3)
    filter_3[2, 2] = -1
    
    # Pass the weights back to the original matrix and return.
    weights[:,:,0,0] = filter_1
    weights[:,:,0,1] = filter_2
    weights[:,:,0,2] = filter_3
    
    return weights

def build():
    # Step_1: define tensorflow model
    # Here a typicall vanilla CNN is defined using three Conv layers followed 
    # by MaxPooling operations and two fully connected layers 
    Net = Sequential([
        # the first parameter defines the #-of feature maps,
        # the second parameter the filter kernel size
        Conv2D(3, (5, 5), padding='same', input_shape=(256, 256, 1)),
        
        Conv2D(96, (7, 7), strides=2, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPool2D(pool_size=(3,3), strides=2, padding='same'),

        Conv2D(64, (5,5), strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPool2D(pool_size=(3,3), strides=2,),

        Conv2D(64, (5,5), strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPool2D(pool_size=(3,3), strides=2,),

        Conv2D(128, (1,1), strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPool2D(pool_size=(3,3), strides=2,),

        Flatten(),
        
        Dense(200),
        Activation('relu'),
        Dense(200),
        Activation('relu'),
        Dense(8, activation='softmax')
    ])
    
    Net.summary()
    return Net