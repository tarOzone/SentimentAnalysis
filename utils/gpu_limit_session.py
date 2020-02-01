from tensorflow.compat.v1 import disable_eager_execution
from tensorflow.compat.v1 import ConfigProto, reset_default_graph, Session
from tensorflow.compat.v1.keras.backend import set_session


class GpuLimitSession:
    def __init__(self, gpu_factor, disable_eager=True, verbose=False):
        self.gpu_factor = gpu_factor
        self.config = ConfigProto()
        self.verbose = verbose
        self.__display("[INFO] Initializing session...")
        # disable eager execution
        if disable_eager:
            disable_eager_execution()

    def __set_gpu_factor(self):
        gpu_options = self.config.gpu_options
        gpu_options.per_process_gpu_memory_fraction = self.gpu_factor
        set_session(Session(config=self.config))
        self.__display(f"[COMPLETE] Session has been initialized with GPU memory factor of {self.gpu_factor}")

    def __clear(self):
        reset_default_graph()
        self.config.Clear()

    def __display(self, msg, **kwargs):
        if self.verbose:
            print(msg, **kwargs)

    def __enter__(self):
        self.__set_gpu_factor()  # GPU memory setting
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__display("[INFO] Clearing session...")
        self.__clear()
        self.__display("[COMPLETE] Session has been cleared.")


if __name__ == "__main__":
    with GpuLimitSession(0.7) as gpu:
        print(gpu)
