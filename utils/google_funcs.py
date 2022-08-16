import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

from utils.funcs import *
from humpback_model_dir import humpback_model

def load_google_new():
    model = humpback_model.Model()
    model.load_weights('models/google_humpback_model')
    model.build((1, 39124, 1))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.BinaryCrossentropy()
    )
    return model

def get_flat_model(model):
    """
    Take nested model from Harvey Matthew and flatten it for ease of use.
    This way trainability of layers can be iteratively defined.

    Args:
        model (tf.keras.Sequential): nested Sequential model from M. Harvey

    Returns:
        tf.keras.Sequential: flat model
    """
    # model_list = []
    # model_list.append(model.layers[0])
    # model_list.append(model.layers[1])
    # model_list.append(model.layers[2]._layers[0])
    
    # for layer in model.layers[2]._layers[1]._layers:
    #   model_list.append(layer)
    
    # # necessary to have unique names
    # model_list[7]._name = 'pool_0'
    
    # # get resnet blocks
    # c = 0
    # for i, high_layer in enumerate(model.layers[2]._layers[2:6]):
    #   for j, layer in enumerate(high_layer._layers):
    #     c+=1
    
    #     for low_layer in layer._residual_path._layers:
    #       model_list.append(low_layer)
    
    #     for low_layer in layer._main_path._layers:
    #       model_list.append(low_layer)
    #     model_list.append(layer._activation)
    
    # # names need to be incremented to make sure every layer name is unique
    # for ind in range(7,len(model_list)):
    #   model_list[ind]._name += f'{ind//9}'
    
    # model_list.append(model.layers[2]._layers[-1])
    # model_list.append(model.layers[-1])
    
    
    model_list = []
    model_list.append(model.layers[0])
    model_list.append(model.layers[1])
    model_list.append(model.layers[2]._layers[0])
    for layer in model.layers[2]._layers[1]._layers:
      model_list.append(layer)
    model_list[7]._name = 'pool_0'
    c = 0
    for i, high_layer in enumerate(model.layers[2]._layers[2:6]):
      for j, layer in enumerate(high_layer._layers):
        c+=1
        model_list.append(layer)
        model_list[7+c]._name += f'_{i}'
    model_list.append(model.layers[2]._layers[-1])
    model_list.append(model.layers[-1])
    
    
    # generate new model
    new_model = tf.keras.Sequential(layers=[layer for layer in model_list])
    new_model.build((1, 39124, 1))
    new_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.BinaryCrossentropy()
    )
    return new_model

def load_google_hub():
    return hub.load('https://tfhub.dev/google/humpback_whale/1')

def load_google_sequential():
    sequential = tf.keras.Sequential([
    tf.keras.layers.Input([39124]),
    tf.keras.layers.Lambda(lambda t: tf.expand_dims(t, -1)),
    hub.KerasLayer(load_google_hub(), trainable=False),
    tf.keras.layers.Activation('sigmoid')
    ])

    sequential.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
    )
    return sequential

def predict_hub(data, model, cntxt_wn_sz, **_):
    preds = []
    for segment in data:
        tensor_sig = tf.convert_to_tensor(segment)
        tensor_sig = tf.expand_dims(tensor_sig, -1)  # makes a batch of size 1
        tensor_sig = tf.expand_dims(tensor_sig, 0)  # makes a batch of size 1
        context_step_samples = tf.cast(cntxt_wn_sz, tf.int64)
        score_fn = model.signatures['score']
        scores = score_fn(waveform=tensor_sig, 
                            context_step_samples=context_step_samples)
        preds.append(scores['scores'].numpy()[0,0,0])
    return np.array(preds)

class GoogleMod():
    def __init__(self, params):
        self.model = get_flat_model(load_google_new())
        # self.model = load_google_sequential()
        self.params = params
        self.params['fmin'] = 0
        self.params['fmax'] = 2250
        
    def load_data(self, file, annots, y_test, y_noise, x_test, x_noise):
        self.file = file
        self.annots = annots
        self.x_test = x_test
        self.x_noise = x_noise
        self.y_test = y_test
        self.y_noise = y_noise
        
    def pred(self, noise = False):
        if noise:
            self.preds_noise = self.model.predict(self.x_noise).T[0]
            return self.preds_noise
        else:
            self.preds_call = self.model.predict(self.x_test).T[0]
            return self.preds_call
    
    def eval(self, noise = False):
        if noise:
            return self.model.evaluate(self.x_noise, self.y_noise)
        else:
            return self.model.evaluate(self.x_test, self.y_test)
    
    def spec(self, num, noise = False):
        if noise:
            prediction = self.preds_noise[num]
            start = self.annots['end'].iloc[-1] + \
                    self.params['cntxt_wn_sz']*num / self.params['sr']
            tensor_sig = tf.convert_to_tensor([self.x_noise[num]])
        else:
            prediction = self.preds_call[num]
            start = self.annots['start'].iloc[num]
            tensor_sig = tf.convert_to_tensor([self.x_test[num]])
        
        tensor_sig = tf.expand_dims(tensor_sig, -1)    
        spec_data = self.model.layers[1].resolved_object.front_end(tensor_sig)
        
        plot_spec(spec_data[0].numpy().T, self.file, prediction = prediction,
                    start = start, noise = noise, 
                    mod_name = type(self).__name__, **self.params)