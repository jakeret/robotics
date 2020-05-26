from typing import List

from tensorflow.keras import Model, Input
from tensorflow.keras import layers


class ResidualBlock(layers.Layer):
    """
    A TCN Residual block stacking the dilated causal convolution
    :param filters: number of output filters in the convolution
    :param kernel_size: length of the 1D convolution window
    :param dilation_rate: dilation rate to use for dilated convolution
    :param dropout_rate: dropout rate
    :param activation: non linearity
    """

    def __init__(self,
                 filters: int,
                 kernel_size: int,
                 dilation_rate: int,
                 dropout_rate: float,
                 activation: str,
                 **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate
        self.activation = activation

        self.causal_conv_1 = layers.Conv1D(filters=self.filters,
                                           kernel_size=kernel_size,
                                           dilation_rate=dilation_rate,
                                           padding='causal')
        self.weight_norm_1 = layers.LayerNormalization()
        self.dropout_1 = layers.SpatialDropout1D(rate=dropout_rate)
        self.activation_1 = layers.Activation(activation)

        self.causal_conv_2 = layers.Conv1D(filters=self.filters,
                                           kernel_size=kernel_size,
                                           dilation_rate=dilation_rate,
                                           padding='causal')
        self.weight_norm_2 = layers.LayerNormalization()
        self.dropout_2 = layers.SpatialDropout1D(rate=dropout_rate)
        self.activation_2 = layers.Activation(activation)

        self.activation_3 = layers.Activation(activation)

    def build(self, input_shape):
        in_channels = input_shape[-1]
        if in_channels == self.filters:
            self.skip_conv = None
        else:
            self.skip_conv = layers.Conv1D(filters=self.filters,
                                           kernel_size=1)

        super(ResidualBlock, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        if self.skip_conv is None:
            skip = inputs
        else:
            skip = self.skip_conv(inputs)

        x = self.causal_conv_1(inputs)
        x = self.weight_norm_1(x)
        x = self.activation_1(x)
        x = self.dropout_1(x, training=training)

        x = self.causal_conv_2(x)
        x = self.weight_norm_2(x)
        x = self.activation_2(x)
        x = self.dropout_2(x, training=training)

        x = self.activation_3(x + skip)
        return x

    def get_config(self):
        return dict(filters=self.filters,
                    kernel_size=self.kernel_size,
                    dilation_rate=self.dilation_rate,
                    dropout_rate=self.dropout_rate,
                    activation=self.activation,
                    **super(ResidualBlock, self).get_config()
                    )


class TCN(layers.Layer):
    """
    The TCN-layer consisting of TCN-residual-blocks.
    The dilation-rate grows exponentially with each residual block.

    :param filters: number of conv filters per residual block
    :param kernel_size: size of the conv kernels
    :param return_sequence: flag if the last sequence should be returned or only last element
    :param dropout_rate: dropout rate, default: 0.0
    :param activation: non linearity, default: relu
    """

    def __init__(self,
                 filters: List[int],
                 kernel_size: int,
                 return_sequence:bool = False,
                 dropout_rate:float = 0.0,
                 activation:str = "relu",
                 **kwargs):

        super(TCN, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.return_sequence = return_sequence
        self.dropout_rate = dropout_rate
        self.activation = activation

        self._blocks = []

        for i in range(self.depth):
            dilation_size = 2 ** i
            self._blocks.append(
                ResidualBlock(filters=filters[i],
                              kernel_size=kernel_size,
                              dilation_rate=dilation_size,
                              dropout_rate=dropout_rate,
                              activation=activation,
                              name=f"residual_block_{i}")
            )

        if not self.return_sequence:
            self.slice_layer = layers.Lambda(lambda tt: tt[:, -1, :])

    def call(self, inputs, training=None, **kwargs):
        x = inputs
        for block in self._blocks:
            x = block(x)

        if not self.return_sequence:
            x = self.slice_layer(x)
        return x

    def get_config(self):
        return dict(filters=self.filters,
                    kernel_size=self.kernel_size,
                    return_sequence=self.return_sequence,
                    dropout_rate=self.dropout_rate,
                    activation=self.activation,
                    **super(TCN, self).get_config()
                    )

    @property
    def depth(self):
        return len(self.filters)

    @property
    def receptive_field_size(self):
        return receptive_field_size(self.kernel_size, self.depth)


def receptive_field_size(kernel_size, depth):
    return 1 + 2 * (kernel_size - 1) * (2 ** depth - 1)


def build_model(sequence_length: int,
                channels: int,
                filters: List[int],
                num_classes:int,
                kernel_size: int,
                return_sequence:bool = False) -> Model:
    """
    Builds a simple TCN model for a classification task

    :param sequence_length: lenght of the input sequence
    :param channels: number of channels of the input sequence
    :param filters: number of conv filters per residual block
    :param num_classes: number of output classes
    :param kernel_size: size of the conv kernels
    :param return_sequence: flag if the last sequence should be returned or only last element

    :return: a tf keras model
    """

    inputs = Input(shape=(sequence_length, channels), name="inputs")
    tcn_block = TCN(filters, kernel_size, return_sequence)
    x = tcn_block(inputs)

    outputs = layers.Dense(num_classes,
                           activation="softmax",
                           name="output")(x)

    model = Model(inputs, outputs, name="tcn")

    print(f"Input sequence lenght: {sequence_length}, model receptive field: {tcn_block.receptive_field_size}")
    return model
