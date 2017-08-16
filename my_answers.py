import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    # containers for input/output pairs
    outputs = []
    inputs = []
    #create list of windows
    for i in range(len(series)-window_size):
        inputs.append([x for x in series[i:i+window_size]])
        outputs.append(series[i+window_size])
    
    #convert lists to numpy arrays
    inputs = np.asarray(inputs)
    outputs = np.asarray(outputs)
    outputs.shape=(len(outputs),1)
    
    
    return inputs, outputs

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(step_size, window_size):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    import keras

    # given - fix random seed - so we can all reproduce the same results on our default time series
    np.random.seed(0)

    # build an RNN to perform regression on our time series input/output data
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size,1)))
    model.add(Dense(1))


    # build model using keras documentation recommended optimizer initialization
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)


### TODO: list all unique characters in the text and remove any non-english ones
def clean_text(text):
    # find all unique characters in the text
    unique = set(text)
    #create list of characters to remove from corpus
    non_english = [c for c in unique if c not in " abcdefghijklmnopqrstuvwxyz:!.;,?'"]
        for r in non_english:
        text = text.replace(r,'')

    # shorten any extra dead space created above
    return text.replace('  ',' ')
 


### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    #initial input string list
    initial_seq = [x for x in text[0:window_size]]
    #create input/outputs from sliding window over text
    for i in range(0,len(text)-window_size,step_size):
        outputs.append(text[i+window_size])
        inputs.append(initial_seq)
        initial_seq = initial_seq[step_size:]
        initial_seq.extend(text[i+window_size:i+window_size+step_size])
        
    return inputs,outputs
