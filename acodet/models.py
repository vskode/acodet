from pathlib import Path
import zipfile
import sys
import torch
import torch.nn as nn
from .torchmodels import TorchModel

import numpy as np

from . import global_config as conf


class ModelHelper:
    """
    Helper class to provide checkpoint loading and changing of input shape.
    """

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
        import tensorflow as tf
        from .humpback_model_dir import front_end
        model_list = self.model.layers
        model_list.insert(0, tf.keras.layers.Input([conf.CONTEXT_WIN]))
        model_list.insert(
            1, tf.keras.layers.Lambda(lambda t: tf.expand_dims(t, -1))
        )
        model_list.insert(2, front_end.MelSpectrogram())
        model_list[4]._name = 'lambda_2'
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
        if kwargs.get('training_path'):
            model_name = kwargs.get('training_path')
            # load specified training
            if Path(f"../trainings/{model_name}/{model_name}.keras").exists():
                self.ckpt=f"../trainings/{model_name}/{model_name}.keras"
            elif Path(f"acodet/src/models/{model_name}/{model_name}.keras").exists():
                self.ckpt=f"acodet/src/models/{model_name}/{model_name}.keras"
            else:
                self.ckpt = None
                print("Advanced config setting `load_ckpt_path` not found")
                return 1
        else:
            self.ckpt = None
            

    def load_model(self, **kwargs):
        from .humpback_model_dir.leaf_pcen import FBetaScore, Support
        import tensorflow as tf
        if not Path(conf.MODEL_DIR).joinpath(conf.MODEL_NAME).exists():
            self.download_model()
            # for model_path in Path(conf.MODEL_DIR).iterdir():
            for model_path in list(Path(conf.MODEL_DIR).glob(conf.MODEL_NAME+'*')):
                if not model_path.suffix == ".zip":
                    continue
                else:
                    with zipfile.ZipFile(model_path, "r") as model_zip:
                        model_zip.extractall(conf.MODEL_DIR)
            
        if '2.15' in tf.__version__:
            self.model = tf.keras.models.load_model(
                Path(conf.MODEL_DIR).joinpath(conf.MODEL_NAME),
                custom_objects={"Addons>FBetaScore": FBetaScore},
            )
        elif int(tf.__version__.split('.')[1]) > 15:
            from acodet.transfer_weights import inject_weights
            from acodet.tf220 import PCEN, Block, ResidualPath, MainPath
            if self.ckpt is None:
                self.model = tf.keras.models.load_model(
                    Path(conf.MODEL_DIR).joinpath(conf.MODEL_NAME+'.keras'),
                    custom_objects={
                        "PCEN": PCEN,
                        "Block": Block,
                        "ResidualPath": ResidualPath,
                        "MainPath": MainPath
                    }
                )
                inject_weights(
                    self.model, 
                    Path(conf.MODEL_DIR) / 'original_model_weights.npz'
                    )
                print(f'Loaded {conf.MODEL_NAME}')
            else:
                self.model = tf.keras.models.load_model(
                    self.ckpt,
                    custom_objects={
                        "PCEN": PCEN,
                        "Block": Block,
                        "ResidualPath": ResidualPath,
                        "MainPath": MainPath,
                        "FBetaScore": FBetaScore,
                        "Support": Support,
                        }
                    )
                print(f'Successfully loaded {self.ckpt}')

    
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
        from .humpback_model_dir import humpback_model
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
        import tensorflow as tf
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
        import tensorflow as tf
        from .humpback_model_dir import leaf_pcen
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
        



def _get_linear_probe_class():
    """
    Return the linear probe class of bacpipe. In bacpipe >= 1.3.2 the class
    is called LinearProbe (with the linear layer `probe`), before it was called
    LinearClassifier (with the linear layer `clfier`). An alias is registered
    for the old name, so that probes that were pickled as complete objects
    with older bacpipe versions can still be unpickled.
    """
    from bacpipe.embedding_evaluation.probing import train_probe

    if hasattr(train_probe, "LinearProbe"):
        if not hasattr(train_probe, "LinearClassifier"):
            train_probe.LinearClassifier = train_probe.LinearProbe
        return train_probe.LinearProbe
    return train_probe.LinearClassifier


def _probe_state_dict_from_checkpoint(checkpoint, target_keys):
    """
    Turn the content of a linear probe checkpoint into a state dict that fits
    the probe of the installed bacpipe version. Checkpoints can be a state
    dict, a bound `state_dict` method or the complete module (depending on
    which acodet/bacpipe version saved them), and their keys can use the old
    `clfier.` or the new `probe.` prefix.
    """
    if isinstance(checkpoint, nn.Module):
        state_dict = checkpoint.state_dict()
    elif callable(checkpoint):
        state_dict = checkpoint()
    else:
        state_dict = checkpoint

    target_prefixes = {k.split(".")[0] for k in target_keys}
    if len(target_prefixes) != 1:
        return state_dict
    target_prefix = next(iter(target_prefixes))

    renamed = {}
    for key, value in state_dict.items():
        prefix, sep, rest = key.partition(".")
        if not sep:
            # state dict of a bare linear layer, e.g. 'weight' and 'bias'
            key = f"{target_prefix}.{key}"
        elif prefix != target_prefix:
            # e.g. 'clfier.weight' (old bacpipe) -> 'probe.weight'
            key = f"{target_prefix}.{rest}"
        renamed[key] = value
    return renamed


class BacpipeModel(nn.Module):
    """
    Wrapper around the bacpipe (>= 1.3.5) Embedder. Either the classifier
    that ships with the bacpipe model is used (bool_lin_clfier: False) or a
    linear classifier that is trained on top of the frozen embeddings
    (bool_lin_clfier: True).
    """

    def __init__(self, num_classes=1, **kwargs):
        super(BacpipeModel, self).__init__()

        import bacpipe
        from bacpipe import settings, Embedder, ensure_models_exist

        if conf.DEVICE == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = conf.DEVICE

        # The checkpoints have to be downloaded to the same directory that
        # the Embedder loads them from.
        if conf.BOOL_BACPIPE_CHCKPTS:
            settings.model_base_path = str(conf.BACPIPE_CHCKPT_DIR)
        ensure_models_exist(
            model_base_path=Path(settings.model_base_path),
            model_names=[conf.MODEL_NAME],
        )

        # bacpipe uses the global batch size to compute the model specific
        # batch size, it has to be an integer
        settings.global_batch_size = max(1, int(conf.BATCH_SIZE // 2))
        settings.device = device
        # Only run the classifier that ships with the bacpipe model if it is
        # used for the predictions. Otherwise its predictions would pile up
        # in the Embedder for every processed file.
        settings.run_pretrained_classifier = not conf.BOOL_LIN_CLFIER

        self.embedder = Embedder(
            model_name=conf.MODEL_NAME,
            **vars(settings)
            )

        conf.SR = self.embedder.model.sr
        conf.CONTEXT_WIN = self.embedder.model.segment_length

        if not conf.BOOL_LIN_CLFIER and not self.embedder.model.bool_classifier:
            raise ValueError(
                f"The bacpipe model {conf.MODEL_NAME} has no classifier. "
                "Set `bool_lin_clfier: True` in the advanced config to use "
                "a linear classifier on top of its embeddings."
            )

        if conf.BOOL_LIN_CLFIER:
            head = _get_linear_probe_class()
            self.lin_classifier = head(
                in_dim=bacpipe.EMBEDDING_DIMENSIONS[conf.MODEL_NAME], out_dim=1
            )

            if conf.MODEL_NAME in bacpipe.TF_MODELS:
                # we need to do this so that the tensorflow model can use
                # the gpu. it will not leave any vram available for torch.
                # therefore we are putting the classifier on the cpu.
                self.probe_device = 'cpu'
            else:
                self.probe_device = device
            self.lin_classifier.to(self.probe_device)

    def load_model(self, training_path):
        if not training_path:
            print(f'\nSuccessfully loaded {conf.MODEL_NAME} without linear classifier\n')
            return

        if Path(f'../trainings/{training_path}/{training_path}_bacpipe_lin_clfier.pt').exists():
            checkpoint_path = Path(f'../trainings/{training_path}/{training_path}_bacpipe_lin_clfier.pt')
        elif Path(f'acodet/src/models/{training_path}/{training_path}_bacpipe_lin_clfier.pt').exists():
            checkpoint_path = Path(f'acodet/src/models/{training_path}/{training_path}_bacpipe_lin_clfier.pt')
        else:
            print(f"Model file {training_path} not found. Please check path.")
            return 1

        if not conf.BOOL_LIN_CLFIER:
            print(
                f"Found linear classifier {checkpoint_path}, but "
                "`bool_lin_clfier` is False in the advanced config, so it is "
                f"not used. Using the classifier of {conf.MODEL_NAME} instead."
            )
            return

        # make sure the old class name exists before unpickling old probes
        _get_linear_probe_class()
        map_location = torch.device(self.probe_device)
        try:
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
        except Exception:
            # older checkpoints contain pickled python objects
            checkpoint = torch.load(
                checkpoint_path, weights_only=False, map_location=map_location
            )

        self.lin_classifier.load_state_dict(
            _probe_state_dict_from_checkpoint(
                checkpoint, self.lin_classifier.state_dict().keys()
            )
        )
        print(f'\nSuccessfully loaded {checkpoint_path}\n')

    def _select_humpback_predictions(self, predictions):
        """Return only the humpback class of the multiclass classifiers."""
        humpback_labels = {
            'perch_v2': 'Megaptera novaeangliae',
            'google_whale': 'Humpback',
        }
        if conf.MODEL_NAME in humpback_labels:
            model_labels = np.array(self.embedder.model.classes)
            humpback_label_idx = np.where(
                model_labels == humpback_labels[conf.MODEL_NAME]
            )[0][0]
            predictions = predictions[:, humpback_label_idx]
        return predictions

    def _embed_tensor(self, x):
        """Compute embeddings for a batch of audio frames (no gradients)."""
        from bacpipe import TF_MODELS

        with torch.no_grad():
            if conf.MODEL_NAME in TF_MODELS:
                x = x.cpu()
            if hasattr(self.embedder.model, "preprocess"):
                x = self.embedder.model.preprocess(x)
            embeds = self.embedder.model(x)
        if not isinstance(embeds, torch.Tensor):
            embeds = torch.as_tensor(np.asarray(embeds))
        embeds = embeds.reshape(len(x), -1) if embeds.dim() > 1 else embeds.unsqueeze(0)
        return embeds

    def forward(self, x, y=None, noise=None, path=None, start=None, training=False, **kwargs):
        if not conf.BOOL_LIN_CLFIER:
            self.embedder.get_embeddings_from_model(x)
            predictions = self.embedder.classifier.predictions
            # the Embedder accumulates the predictions of the built-in
            # classifier, reset them so that the next file starts empty
            self.embedder.classifier.predictions = torch.tensor([])
            return self._select_humpback_predictions(predictions)
        else:
            if isinstance(x, torch.Tensor):
                embeds = self._embed_tensor(x)
            else:
                with torch.no_grad():
                    embeds = self.embedder.get_embeddings_from_model(x)
                embeds = torch.as_tensor(np.asarray(embeds))

            embeds = embeds.to(self.probe_device, dtype=torch.float32)
            logits = self.lin_classifier(embeds)
            return logits


def init_model(
    model_name: str = conf.MODELCLASSNAME,
    training_path: str = conf.LOAD_CKPT_PATH,
    input_specs=False,
    **kwargs,
    ):
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
    mod_obj = model_class(training_path=training_path, **kwargs)
    mod_obj.load_model(training_path=training_path)
    if conf.MODEL_NAME in ["FlatHBNA"] or conf.MODELCLASSNAME == "BacpipeModel":
        input_specs = True
    if not input_specs and hasattr(mod_obj, 'change_input_to_array'):
        mod_obj.change_input_to_array()
    if conf.MODELCLASSNAME == 'HumpBackNorthAtlantic':
        return mod_obj.model
    else:
        return mod_obj
