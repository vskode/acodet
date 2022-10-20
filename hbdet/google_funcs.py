import tensorflow as tf
import yaml
with open('hbdet/hbdet/config.yml', 'r') as f:
    config = yaml.safe_load(f)

from .humpback_model_dir import humpback_model

class GoogleMod():
    def __init__(self, **params) -> None:
        """
        This class is the framework to load and flatten the model created
        by Matthew Harvey in collaboration with Ann Allen for the 
        PIFSC HARP data 
        (https://www.frontiersin.org/article/10.3389/fmars.2021.607321). 

        Args:
            params (dict): model parameters
        """
        self.load_google_new(**params)
        self.load_flat_model(**params)
    
    def load_google_new(self, load_g_ckpt=True, **_):
        """
        Load google model architecture. By default the model weights are 
        initiated with the pretrained weights from the google checkpoint. 

        Args:
            load_g_ckpt (bool, optional): Initialize model weights with Google
            pretrained weights. Defaults to True.
        """
        self.model = humpback_model.Model()
        if load_g_ckpt:
            self.model.load_weights('models/google_humpback_model')

    def load_flat_model(self, input_tensors='spectrograms', **_):
        """
        Take nested model architecture from Harvey Matthew and flatten it for 
        ease of use. This way trainability of layers can be iteratively 
        defined. The model still has a nested structure. The ResNet blocks are
        combined into layers of type Block, but because their trainability would 
        only be changed on the block level, this degree of nesting shouldn't
        complicate the usage of the model.
        By default the model is itiated with spectrograms of shape [128, 64] as
        inputs. This means that spectrograms have to be precomputed. 
        Alternatively if the argument input_tensors is set to something else, 
        inputs are assumed to be audio arrays of 39124 samples length.
        As this process is very specific to the model ascertained from 
        Mr. Harvey, layer indices are hard coded.

        Args:
            input_tensors (str): input type, if not spectrograms, they are 
            assumed to be audio arrays of 39124 samples length. 
            Defaults to 'spectrograms'.
        """
        # create list which will contain all the layers
        model_list = []
        if input_tensors == 'spectrograms':
            # add Input layer
            model_list.append(tf.keras.layers.Input([128, 64]))
        else:
            # add MelSpectrogram layer
            model_list.append(tf.keras.layers.Input(
                [config['cntxt_wn_sz']]))
            model_list.append(tf.keras.layers.Lambda(
                lambda t: tf.expand_dims(t, -1)))
            model_list.append(self.model.layers[0])
        # add PCEN layer
        model_list.append(self.model.layers[1])
        # iterate through PreBlocks
        model_list.append(self.model.layers[2]._layers[0])
        for layer in self.model.layers[2]._layers[1]._layers:
            model_list.append(layer)
        # change name, to make sure every layer has a unique name
        num_preproc_layers = len(model_list)
        model_list[num_preproc_layers-1]._name = 'pool_pre_resnet'
        # iterate over ResNet blocks
        c = 0
        for i, high_layer in enumerate(self.model.layers[2]._layers[2:6]):
            for j, layer in enumerate(high_layer._layers):
                c+=1
                model_list.append(layer)
                model_list[num_preproc_layers-1+c]._name += f'_{i}'
        # add final Dense layers
        model_list.append(self.model.layers[2]._layers[-1])
        model_list.append(self.model.layers[-1])
        # normalize results between 0 and 1
        model_list.append(tf.keras.layers.Activation('sigmoid'))
        
        # generate new model
        self.model = tf.keras.Sequential(
            layers=[layer for layer in model_list])
