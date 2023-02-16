import tensorflow as tf
from pathlib import Path
import zipfile
import sys

from hbdet.funcs import get_val_labels
from . import global_config as conf
from .humpback_model_dir import humpback_model
from .humpback_model_dir import front_end
from .humpback_model_dir import leaf_pcen

class ModelHelper:    
    def load_ckpt(self, ckpt_path, ckpt_name='last'):
        if isinstance(ckpt_path, Path):
            ckpt_path = ckpt_path.stem
        ckpt_path = (Path('../trainings').joinpath(ckpt_path)
                     .joinpath('unfreeze_no-TF')) # TODO namen ändern
        try:
            file_path = (ckpt_path.joinpath(f'cp-{ckpt_name}.ckpt.index'))
            if not file_path.exists():
                ckpts = list(ckpt_path.glob('cp-*.ckpt*'))
                ckpts.sort()
                ckpt = ckpts[-1]
            else:
                ckpt = file_path
            self.model.load_weights(
                str(ckpt).replace('.index', '')
                ).expect_partial()
        except Exception as e:
            print('Checkpoint not found.', e)

    def change_input_to_array(self):
        """
        change input layers of model after loading checkpoint so that a file
        can be predicted based on arrays rather than spectrograms, i.e.
        reintegrate the spectrogram creation into the model. 

        Args:
            model (tf.keras.Sequential): keras model

        Returns:
            tf.keras.Sequential: model with new arrays as inputs
        """
        model_list = self.model.layers
        model_list.insert(0, tf.keras.layers.Input([conf.CONTEXT_WIN]))
        model_list.insert(1, tf.keras.layers.Lambda(
                            lambda t: tf.expand_dims(t, -1)))
        model_list.insert(2, front_end.MelSpectrogram())
        self.model = tf.keras.Sequential(layers=[layer for layer in model_list])

class HumpBackNorthAtlantic(ModelHelper):
    def __init__(self, **kwargs):
        pass

    def load_model(self, **kwargs):
        if not Path(conf.MODEL_DIR).joinpath(conf.MODEL_NAME).exists():
            for model_path in Path(conf.MODEL_DIR).iterdir():
                if not model_path.suffix == '.zip':
                    continue
                else:
                    with zipfile.ZipFile(model_path, 'r') as model_zip:
                        model_zip.extractall(conf.MODEL_DIR)
        self.model = tf.keras.models.load_model(Path(conf.MODEL_DIR)
                                   .joinpath(conf.MODEL_NAME))

class GoogleMod(ModelHelper): # TODO change name
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
    
    def load_google_new(self, load_g_ckpt=conf.LOAD_G_CKPT, **_):
        """
        Load google model architecture. By default the model weights are 
        initiated with the pretrained weights from the google checkpoint. 

        Args:
            load_g_ckpt (bool, optional): Initialize model weights with Google
            pretrained weights. Defaults to True.
        """
        self.model = humpback_model.Model()
        if load_g_ckpt:
            self.model = self.model.load_from_tf_hub()

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
                [conf.CONTEXT_WIN]))
            model_list.append(tf.keras.layers.Lambda(
                lambda t: tf.expand_dims(t, -1)
                ))
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

class KerasAppModel(ModelHelper):
    def __init__(self, keras_mod_name='EfficientNetB0', **params) -> None:
        keras_model = getattr(tf.keras.applications, keras_mod_name)(
                include_top=True,
                weights=None,
                input_tensor=None,
                input_shape=[128, 64, 3],
                pooling=None,
                classes=1,
                classifier_activation="sigmoid"
            )
        
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input([128, 64]),
            leaf_pcen.PCEN(
                alpha=0.98,
                delta=2.0,
                root=2.0,
                smooth_coef=0.025,
                floor=1e-6,
                trainable=True,
                name='pcen',
            ),
            # tf.keras.layers.Lambda(lambda t: 255. *t /tf.math.reduce_max(t)),
            tf.keras.layers.Lambda(lambda t: tf.tile(
                tf.expand_dims(t, -1),
                [1 for _ in range(3)] + [3])),
            keras_model
        ])
        

def init_model(model_name: str =conf.MODELCLASSNAME, 
               training_path: str =conf.LOAD_CKPT_PATH, 
               input_specs=False, **kwargs) -> tf.keras.Sequential:
    """
    Initiate model instance, load weights. As the model is trained on 
    spectrogram tensors but will now be used for inference on audio files
    containing continuous audio arrays, the input shape of the model is 
    changed after the model weights have been loaded. 

    Parameters
    ----------
    model_instance : type
        callable class to create model object
    training_path : str
        checkpoint path

    Returns
    -------
    tf.keras.Sequential
        the sequential model with pretrained weights
    """
    model_class = getattr(sys.modules[__name__], model_name)
    mod_obj = model_class(**kwargs)
    if conf.MODEL_NAME == 'FlatHBNA':
        input_specs = True
    if model_class == HumpBackNorthAtlantic:
        mod_obj.load_model()
    else:
        mod_obj.load_ckpt(training_path)
    if not input_specs:
        mod_obj.change_input_to_array()
    return mod_obj.model

def get_labels_and_preds(model_name: str, 
                         training_path: str, 
                         val_data: tf.data.Dataset, 
                         **kwArgs) -> tuple:
    """
    Retrieve labels and predictions of validation set and given model
    checkpoint. 

    Parameters
    ----------
    model_instance : type
        model class
    training_path : str
        path to checkpoint
    val_data : tf.data.Dataset
        validation dataset

    Returns
    -------
    labels: np.ndarray
        labels
    preds: mp.ndarray
        predictions
    """
    model = init_model(load_from_ckpt=True, model_name=model_name, 
                       training_path=training_path, **kwArgs)
    preds = model.predict(x = prep_ds_4_preds(val_data))
    labels = get_val_labels(val_data, len(preds))
    return labels, preds

def prep_ds_4_preds(dataset):
    """
    Prepare dataset for prediction, by batching and ensuring that only
    arrays and corresponding labels are passed (necessary because if 
    data about the origin of the array is passed, i.e. the file path and
    start time of the array within that file, model.predict fails.).

    Parameters
    ----------
    dataset : TfRecordDataset
        dataset

    Returns
    -------
    TFRecordDataset
        prepared dataset
    """
    if len(list(dataset.take(1))[0]) > 2:
        return dataset.map(lambda x, y, z, w: (x, y)).batch(batch_size=32)
    else:
        return dataset.batch(batch_size=32)