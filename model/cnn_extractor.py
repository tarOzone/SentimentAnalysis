from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, GlobalMaxPooling2D


class CnnExtractor:

    def __init__(self, input_shape, trainable):
        # assertion step
        asset_msg = "Input size must be tuple of two integers (e.g. (800,600))"
        assert type(input_shape) == tuple, asset_msg
        assert type(input_shape[0]) == int and type(input_shape[1]) == int, asset_msg

        # build the model
        inputs = Input(shape=input_shape)   # set Input
        base_model = self._init_base_model(inputs)  # init base model (default as InceptionV3)
        output = self._init_output(base_model.output)   # then flatten as output

        # set trainable the layers in transfer learning
        self.model = Model(base_model.inputs, output)
        self._freeze_model(self.model, trainable)

    def extract(self, x):
        return self.model.predict(x)

    def _freeze_model(self, model, trainable):
        for layer in model.layers:
            layer.trainable = trainable

    def _init_output(self, output):
        x = GlobalMaxPooling2D()(output)
        return x

    def _init_base_model(self, inputs):
        from tensorflow.keras.applications.inception_v3 import InceptionV3
        return InceptionV3(include_top=False, weights="imagenet", input_tensor=inputs)
