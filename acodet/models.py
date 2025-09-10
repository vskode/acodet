import tensorflow as tf
# from tensorflow_addons import metrics
from pathlib import Path
import zipfile
import sys
import json

import numpy as np

from acodet.funcs import get_val_labels
from . import global_config as conf
from .humpback_model_dir import humpback_model
from .humpback_model_dir import front_end
from .humpback_model_dir import leaf_pcen


class ModelHelper:
    """
    Helper class to provide checkpoint loading and changing of input shape.
    """

    def load_ckpt(self, ckpt_path, ckpt_name="last"):
        if isinstance(ckpt_path, Path):
            ckpt_path = ckpt_path.stem
        ckpt_path = (
            Path("../trainings").joinpath(ckpt_path).joinpath(f"unfreeze_{conf.UNFREEZE}")
        )  # TODO namen ändern
        try:
            file_path = ckpt_path.joinpath(f"cp-{ckpt_name}.ckpt.index")
            if not file_path.exists():
                ckpts = list(ckpt_path.glob("cp-*.ckpt*"))
                ckpts.sort()
                ckpt = ckpts[-1]
            else:
                ckpt = file_path
            self.model.load_weights(
                str(ckpt).replace(".index", "")
            ).expect_partial()
        except Exception as e:
            print("Checkpoint not found.", e)

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
        model_list.insert(
            1, tf.keras.layers.Lambda(lambda t: tf.expand_dims(t, -1))
        )
        model_list.insert(2, front_end.MelSpectrogram())
        for i, layer in enumerate(model_list):
            if 'lambda' in layer.name:
                layer._name = f'lambda_{i}'
        self.model = tf.keras.Sequential(
            layers=model_list
        )

class HumpBackNorthAtlantic(ModelHelper):
    """
    Defualt class for North Atlantic Humpback Whale Song detection. If no new
    training is performed this class is the default class. The model is
    currently included in the repository. The model will be extracted
    and made ready for use.

    Parameters
    ----------
    ModelHelper : class
        helper class providing necessary functionalities
    """

    def __init__(self, **kwargs):
        pass

    def load_model(self, **kwargs):
        if not Path(conf.MODEL_DIR).joinpath(conf.MODEL_NAME).exists():
            self.download_model()
            # for model_path in Path(conf.MODEL_DIR).iterdir():
            for model_path in list(Path(conf.MODEL_DIR).glob(conf.MODEL_NAME+'*')):
                if not model_path.suffix == ".zip":
                    continue
                else:
                    with zipfile.ZipFile(model_path, "r") as model_zip:
                        model_zip.extractall(conf.MODEL_DIR)
            
        self.model = tf.keras.models.load_model(
            Path(conf.MODEL_DIR).joinpath(conf.MODEL_NAME),
            custom_objects={"Addons>FBetaScore": FBetaScore},
        )
    
    def download_model(self):
        import gdown
        g_drive_link = (
            'https://drive.google.com/uc?id=1qAqAy_REaIqgVM1O5qsNQIBNB8Hb0spz'
            )
        Path(conf.MODEL_DIR).mkdir(parents=True, exist_ok=True)
        output = Path(conf.MODEL_DIR).joinpath(conf.MODEL_NAME + '.zip')  # Change this to your preferred filename
        gdown.download(g_drive_link, str(output), quiet=False)

        print(f"File downloaded as {output}")


class GoogleMod(ModelHelper):  # TODO change name
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

    def load_flat_model(self, input_tensors="spectrograms", **_):
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
        if input_tensors == "spectrograms":
            # add Input layer
            model_list.append(tf.keras.layers.Input([128, 64]))
        else:
            # add MelSpectrogram layer
            model_list.append(tf.keras.layers.Input([conf.CONTEXT_WIN]))
            model_list.append(
                tf.keras.layers.Lambda(lambda t: tf.expand_dims(t, -1))
            )
            model_list.append(self.model.layers[0])

        # add PCEN layer
        model_list.append(self.model.layers[1])
        # iterate through PreBlocks
        model_list.append(self.model.layers[2]._layers[0])
        for layer in self.model.layers[2]._layers[1]._layers:
            model_list.append(layer)
        # change name, to make sure every layer has a unique name
        num_preproc_layers = len(model_list)
        model_list[num_preproc_layers - 1]._name = "pool_pre_resnet"
        # iterate over ResNet blocks
        c = 0
        for i, high_layer in enumerate(self.model.layers[2]._layers[2:6]):
            for j, layer in enumerate(high_layer._layers):
                c += 1
                model_list.append(layer)
                model_list[num_preproc_layers - 1 + c]._name += f"_{i}"
        # add final Dense layers
        model_list.append(self.model.layers[2]._layers[-1])
        model_list.append(self.model.layers[-1])
        # normalize results between 0 and 1
        model_list.append(tf.keras.layers.Activation("sigmoid"))

        # generate new model
        self.model = tf.keras.Sequential(
            layers=[layer for layer in model_list]
        )


class KerasAppModel(ModelHelper):
    """
    Class providing functionalities for usage of standard keras application
    models like EfficientNet. The keras application name is passed and
    the helper class is then used in case existing checkpoints need to be
    loaded or the shape of the input array needs change.

    Parameters
    ----------
    ModelHelper : class
        helper class providing necessary functionalities
    """

    def __init__(self, keras_mod_name=conf.KERAS_MOD_NAME, **params) -> None:
        if Path(conf.MODEL_DIR).joinpath(conf.MODEL_NAME).exists():
            self.model = tf.keras.models.load_model(
                Path(conf.MODEL_DIR).joinpath(conf.MODEL_NAME),
                # custom_objects={"FBetaScote": metrics.FBetaScore},
        )
            if conf.MODEL_NAME == 'birdnet':
                self.model = self.model.model
                conf.SR = 48000
                conf.CONTEXT_WIN = 144000
        else:
            keras_model = getattr(tf.keras.applications, keras_mod_name)(
                include_top=False,
                weights=None,
                input_tensor=None,
                input_shape=[128, 64, 3],
                pooling="avg",
                # classes=1, # steckt jetzt im dense layer
                classifier_activation="sigmoid",
            )
        
            preprocess_fn = tf.keras.applications.efficientnet_v2.preprocess_input

            self.model = tf.keras.Sequential(
                [
                    tf.keras.layers.Input([128, 64]),
                    leaf_pcen.PCEN(
                        alpha=0.98,
                        delta=2.0,
                        root=2.0,
                        smooth_coef=0.025,
                        floor=1e-6,
                        trainable=True,
                        name="pcen",
                    ),
                    # tf.keras.layers.Lambda(lambda t: 255. *t /tf.math.reduce_max(t)),
                    # tf.keras.layers.Lambda(lambda t: (t - tf.reduce_min(t)) / (tf.reduce_max(t) - tf.reduce_min(t) + 1e-6)),  # Normalize PCEN output
                    tf.keras.layers.Lambda(
                        lambda t: tf.tile(
                            tf.expand_dims(t, -1), [1 for _ in range(3)] + [3]
                        )
                    ),
                    tf.keras.layers.Lambda(preprocess_fn),  # Apply preprocessing here
                    keras_model,
                    tf.keras.layers.Dense(1, activation="sigmoid"),  # Add final classifier layer explicitly
                ]
            )


class BacpipeModel:
    def __init__(self, **kwargs):
        import torch
        from bacpipe import config, settings
        from bacpipe.embedding_evaluation.classification.train_classifier import LinearClassifier
        from bacpipe.generate_embeddings import Embedder
        config.models = [conf.MODEL_NAME]
        settings.global_batch_size = conf.BATCH_SIZE
        if conf.DEVICE == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = conf.DEVICE
        if conf.BOOL_BACPIPE_CHPNTS:
            settings.model_base_path = conf.BACPIPE_CHPNT_DIR
            
        settings.device = device
        self.device = 'cpu'
        self.model = Embedder(model_name=conf.MODEL_NAME, **vars(settings))
        
        conf.SR = self.model.model.sr
        conf.CONTEXT_WIN = self.model.model.segment_length
        
        if conf.LINEAR_CLFIER_BOOL:
            
            clfier = torch.load(Path(conf.LIN_CLFIER_DIR) / 'linear_classifier.pt')
            with open(Path(conf.LIN_CLFIER_DIR) / 'label2index.json', 'r') as f:
                label2index = json.load(f)
            self.clfier = LinearClassifier(clfier['clfier.weight'].shape[-1], len(label2index))
            self.clfier.load_state_dict(clfier)
            self.clfier.to(self.device)
            self.model.model.classes = list(label2index.keys())

        self.model.classify = self.classify
            
    def classify(self, file, **kwargs):
        if 'progbar1' in kwargs:
            callback = (lambda frac: (kwargs['progbar1']
                                      .progress(frac, text='Current File')))
            
        import torch
        frames = self.model.prepare_audio(file)
        batched_frames = self.model.model.init_dataloader(frames)
        embeds = self.model.model.batch_inference(batched_frames, callback=callback)
        
        if conf.LINEAR_CLFIER_BOOL:
            logits = self.clfier(embeds.to(self.device))
            predictions = torch.nn.functional.softmax(logits, dim=0)
        else:
            predictions = self.model.model.classifier_outputs[-embeds.shape[0]:]
            
        return predictions.detach().numpy()
        

def init_model(
    model_name: str = conf.MODELCLASSNAME,
    training_path: str = conf.LOAD_CKPT_PATH,
    input_specs=False,
    **kwargs,
) -> tf.keras.Sequential:
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
    if conf.MODEL_NAME in ["FlatHBNA"] or conf.MODELCLASSNAME == "BacpipeModel":
        input_specs = True
    if model_class == HumpBackNorthAtlantic:
        mod_obj.load_model()
    elif training_path:
        mod_obj.load_ckpt(training_path)
    if not input_specs:
        mod_obj.change_input_to_array()
    return mod_obj.model


def get_labels_and_preds(
    model_name: str, training_path: str, val_data: tf.data.Dataset, **kwArgs
) -> tuple:
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
    model = init_model(
        load_from_ckpt=True,
        model_name=model_name,
        training_path=training_path,
        **kwArgs,
    )
    preds = model.predict(x=prep_ds_4_preds(val_data))
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

class FBetaScore(tf.keras.metrics.Metric):
    def __init__(self, num_classes=1, average=None, beta=0.5, threshold=0.5, name="fbeta", dtype=tf.float32, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.num_classes = num_classes
        self.average = average
        self.beta = beta
        self.threshold = threshold

        # Must match variable names used in TFA version
        self.true_positives = self.add_weight(name="true_positives", shape=(num_classes,), initializer="zeros")
        self.false_positives = self.add_weight(name="false_positives", shape=(num_classes,), initializer="zeros")
        self.false_negatives = self.add_weight(name="false_negatives", shape=(num_classes,), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > self.threshold, self.dtype)
        y_true = tf.cast(y_true, self.dtype)

        tp = tf.reduce_sum(y_true * y_pred, axis=0)
        fp = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
        fn = tf.reduce_sum(y_true * (1 - y_pred), axis=0)

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        beta_sq = self.beta ** 2
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-7)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-7)

        return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall + 1e-7)

    def reset_state(self):
        for v in self.variables:
            v.assign(tf.zeros_like(v))

    def get_config(self):
        return {
            "num_classes": self.num_classes,
            "average": self.average,
            "beta": self.beta,
            "threshold": self.threshold,
        }
