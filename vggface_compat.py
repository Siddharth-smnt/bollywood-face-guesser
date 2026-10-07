"""Import compatibility for unmodified keras-vggface 0.6 on Keras 2.15.

No model layers, weights, or preprocessing functions are replaced.
Only the TensorFlow, channels-last ResNet50 path is supported.
"""

import sys
import types


def enable_legacy_imports():
    import keras

    if not keras.__version__.startswith("2.15."):
        raise RuntimeError(
            "This deployment uses Keras 2.15. Install requirements.txt in a "
            "separate Python 3.11 environment."
        )

    from keras.src.utils import layer_utils

    # keras-vggface imports three module locations removed from newer Keras.
    keras.utils.layer_utils = layer_utils
    sys.modules.setdefault("keras.utils.layer_utils", layer_utils)

    data_utils = types.ModuleType("keras.utils.data_utils")
    data_utils.get_file = keras.utils.get_file
    sys.modules.setdefault("keras.utils.data_utils", data_utils)

    engine = types.ModuleType("keras.engine")
    engine.__path__ = []
    topology = types.ModuleType("keras.engine.topology")
    topology.get_source_inputs = keras.utils.get_source_inputs
    engine.topology = topology
    sys.modules.setdefault("keras.engine", engine)
    sys.modules.setdefault("keras.engine.topology", topology)

    keras.backend.set_image_data_format("channels_last")

