"""
    define ANN models (architecture + optimization params) using Keras

    Build functions in this module should:

    - take an argument, fp_size, defining the size of input vectors
    - Using keras, define the model architecture 
        - via either Sequential or Functional API
    - call compile() method on model with optimizer and loss function defined
    - return the model instance

    The keras.Model instance that is returned can be trained following the steps
    outlined in the user guide (https://keras.io/guides/training_with_built_in_methods/)
"""


from tensorflow import keras
from keras.layers import Activation, Dense
from tensorflow.keras.layers import Dropout
from keras.callbacks import EarlyStopping



def build_linear_model(fp_size: int,
                       ) -> keras.Model :
    """
    build a simple linear model with architecture:

    - [input] (fp_size,)
    - [output] <1 node>

    and optimization params:
    
    - loss: MSE
    - optimizer = RMSProp

    Parameters
    ----------
    fp_size : ``int``
        size of molecular fingerprints (bits)

    Returns
    -------
    model : ``keras.Model``
        keras model (Sequential), compiled
    """
    model = keras.Sequential()
    model.add(Dense(64, activation='relu', input_shape=(fp_size,))) 
    model.add(Dense(128, activation='relu'))  
    model.add(Dense(64, activation='relu'))  
    model.add(Dense(32, activation='linear'))  
    model.add(Dense(1, activation='linear'))  

    model.compile(optimizer=keras.optimizers.Adam(), 
                  loss=keras.losses.MeanSquaredError(), 
                  metrics=[
                      keras.metrics.RootMeanSquaredError(),
                      keras.metrics.MeanAbsoluteError(),
                  ])
    #es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

    return model

