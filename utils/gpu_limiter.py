import tensorflow as tf
from tensorflow.compat.v1 import disable_eager_execution
from tensorflow.compat.v1 import ConfigProto, reset_default_graph


class Session:
    def __init__(self, gpu_factor, disable_eager=True):
        self.display("[INFO] Setting session...")
        self.config = ConfigProto()
        # GPU memory setting
        self._set_gpu_factor(self.config, gpu_factor)
        # disable eager execution
        if disable_eager:
            disable_eager_execution()

    def _set_gpu_factor(self, config, gpu_factor):
        config.gpu_options.per_process_gpu_memory_fraction = gpu_factor
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
        self.display(f"[COMPLETE] Session has been set with GPU mamory factor {gpu_factor}")

    def display(self, msg, **kwargs):
        print(msg, **kwargs)

    def __del__(self):
        self.display("[INFO] Clearing session...")
        reset_default_graph()
        self.display("[COMPLETE] Session has been cleared.")