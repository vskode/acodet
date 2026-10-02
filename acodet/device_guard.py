"""Guard against a TensorFlow / cuDNN version mismatch.

TensorFlow 2.20 (Keras 3) is compiled against cuDNN 9.3. The ``bacpipe`` and
``torch`` dependencies pin ``nvidia-cudnn-cu12==9.1``, so when they are
installed alongside TensorFlow 2.20 the runtime ends up loading cuDNN 9.1.
TensorFlow still enumerates a GPU in that case, but the first convolution fails
with::

    Loaded runtime CuDNN library: 9.1.0 but source was compiled with: 9.3.0
    ...
    INVALID_ARGUMENT: No DNN in stream executor.

The only reliable fix is to make TensorFlow ignore the GPU entirely, which must
happen *before* TensorFlow is first imported. Importing this module runs the
check immediately and, whenever cuDNN is not 9.3, sets
``CUDA_VISIBLE_DEVICES=-1``.
"""

import logging
import os

logger = logging.getLogger(__name__)

# cuDNN version the Keras 3 TensorFlow wheels (>= 2.16) are built against.
REQUIRED_CUDNN_MAJOR = 9
REQUIRED_CUDNN_MINOR = 3


def _parse_dotted_version(raw):
    """Parse a dotted version string (e.g. ``"9.1.0.70"``) into ``(major, minor)``."""
    try:
        parts = str(raw).split(".")
        return int(parts[0]), int(parts[1])
    except (ValueError, IndexError, AttributeError):
        return None


def _cudnn_version_from_pip():
    """Return ``(major, minor)`` for the ``nvidia-cudnn-cu12`` package, if present."""
    try:
        from importlib.metadata import version

        return _parse_dotted_version(version("nvidia-cudnn-cu12"))
    except Exception:
        return None


def _cudnn_version_from_torch():
    """Return ``(major, minor)`` as reported by PyTorch, if available."""
    try:
        import torch

        raw = torch.backends.cudnn.version()
    except Exception:
        return None
    if not raw:
        return None

    # cuDNN 9 encodes the version as ``major * 10000 + minor * 100 + patch``
    # (e.g. 90300 for 9.3.0); cuDNN 8 and earlier use ``major * 1000 + ...``.
    if raw >= 10000:
        return raw // 10000, (raw // 100) % 100
    return raw // 1000, (raw // 100) % 10


def _detect_cudnn_version():
    """Return ``(major, minor)`` for the installed cuDNN, or ``None``."""
    for probe in (_cudnn_version_from_pip, _cudnn_version_from_torch):
        detected = probe()
        if detected is not None:
            return detected
    return None


def cudnn_is_compatible():
    """Return ``True`` when the installed cuDNN matches the required version.

    If the version cannot be determined at all, ``True`` is returned so a GPU
    (if present) is not disabled without evidence of a mismatch.
    """
    detected = _detect_cudnn_version()
    if detected is None:
        return True
    major, minor = detected
    return major == REQUIRED_CUDNN_MAJOR and minor == REQUIRED_CUDNN_MINOR


def enforce_cpu_if_cudnn_incompatible():
    """Force TensorFlow to use the CPU when cuDNN is not the required version.

    This must be called before ``import tensorflow`` (or ``import keras`` with
    the TensorFlow backend) so that TensorFlow never registers a GPU device.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        return  # The user already forced CPU; nothing to do.

    detected = _detect_cudnn_version()
    if detected is None:
        return  # Unknown cuDNN; leave device selection to TensorFlow.

    major, minor = detected
    if major == REQUIRED_CUDNN_MAJOR and minor == REQUIRED_CUDNN_MINOR:
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    logger.warning(
        "Detected cuDNN %d.%d, but TensorFlow >= 2.16 requires cuDNN %d.%d. "
        "Forcing the CPU (CUDA_VISIBLE_DEVICES=-1) to avoid 'No DNN in stream "
        "executor' errors.",
        major,
        minor,
        REQUIRED_CUDNN_MAJOR,
        REQUIRED_CUDNN_MINOR,
    )


# Run the guard as soon as this module is imported.
enforce_cpu_if_cudnn_incompatible()
