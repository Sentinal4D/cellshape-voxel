from torch import nn

from .encoders.resnet import generate_model
from .encoders.convolutional import ConvolutionalEncoder
from .decoders.trans_convolutional import ConvolutionalDecoder


class AutoEncoder(nn.Module):
    def __init__(
        self,
        num_layers_encoder=3,
        num_layers_decoder=3,
        encoder_type='simple',
        input_shape=(64, 64, 64, 1),
        filters=(32, 64, 128, 256, 512),
        num_features=50,
        bias=True,
        activations=False,
        batch_norm=True,
        leaky=True,
        neg_slope=0.01,
        resnet_depth=10
    ):
        super(AutoEncoder, self).__init__()
        assert ((encoder_type == 'simple') or (encoder_type == 'resnet'))
        self.num_layers_encoder = num_layers_encoder
        self.num_layers_decoder = num_layers_decoder
        self.encoder_type = encoder_type
        self.input_shape = input_shape
        self.filters = filters
        self.num_features = num_features
        self.bias = bias
        self.activations = activations
        self.batch_norm = batch_norm
        self.leaky = leaky
        self.neg_slope = neg_slope
        self.resnet_depth = resnet_depth

        if encoder_type == 'simple':
            self.encoder = ConvolutionalEncoder(num_layers_encoder,
                                                input_shape,
                                                       filters,
                                                       num_features,
                                                       bias,
                                                       activations,
                                                       batch_norm,
                                                       leaky,
                                                       neg_slope)
        else:
            self.encoder = generate_model(resnet_depth)

        self.decoder = ConvolutionalDecoder(num_layers_decoder,
                                                input_shape,
                                                       filters,
                                                       num_features,
                                                       bias,
                                                       activations,
                                                       batch_norm,
                                                       leaky,
                                                       neg_slope)

    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
        return output, features


