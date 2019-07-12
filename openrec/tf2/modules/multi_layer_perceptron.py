import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

def MLP(units_list, use_bias=True, activation='relu', out_activation=None):
    
    mlp = Sequential()
    
    for units in units_list[:-1]:
        mlp.add(Dense(units, 
                        activation=activation, 
                        use_bias=use_bias))
    
    mlp.add(Dense(units_list[-1], 
                activation=out_activation, 
                use_bias=use_bias))
    
    return mlp