import torch.nn as nn
import copy


class ConvolutionalDecoder(nn.Module):
    def __init__(
        self,
        num_layers=3,
        input_shape=(64, 64, 64, 1),
        filters=(32, 64, 128, 256, 512),
        num_features=50,
        bias=True,
        activations=False,
        batch_norm=True,
        leaky=True,
        neg_slope=0.01,
    ):
        super(ConvolutionalDecoder, self).__init__()
        assert (
            (num_layers == 3) or (num_layers == 4) or (num_layers == 5)
        ), "Please choose number of layers to be 3, 4, or 5"

        self.num_layers = num_layers
        self.input_shape = input_shape
        self.filters = filters
        self.num_features = num_features
        self.bias = bias
        self.activations = activations
        self.batch_norm = batch_norm
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=neg_slope)
        else:
            self.relu = nn.ReLU(inplace=False)

        self.relu1 = copy.deepcopy(self.relu)
        self.relu2 = copy.deepcopy(self.relu)
        self.relu3 = copy.deepcopy(self.relu)
        self.relu4 = copy.deepcopy(self.relu)
        self.relu5 = copy.deepcopy(self.relu)

        out_pad3 = 1 if input_shape[0] // 2 // 2 % 2 == 0 else 0
        out_pad2 = 1 if input_shape[0] // 2 % 2 == 0 else 0
        out_pad1 = 1 if input_shape[0] % 2 == 0 else 0

        if num_layers == 3:
            lin_features_len = (
                ((self.input_shape[0] // 2 // 2 - 1) // 2)
                * ((self.input_shape[1] // 2 // 2 - 1) // 2)
                * ((self.input_shape[2] // 2 // 2 - 1) // 2)
                * filters[2]
            )
            self.deconv3 = nn.ConvTranspose3d(
                filters[2],
                filters[1],
                3,
                stride=2,
                padding=0,
                output_padding=out_pad3,
                bias=bias,
            )
        else:
            self.deconv3 = nn.ConvTranspose3d(
                filters[2],
                filters[1],
                5,
                stride=2,
                padding=2,
                output_padding=out_pad3,
                bias=bias,
            )

        if num_layers == 4:
            lin_features_len = (
                ((self.input_shape[0] // 2 // 2 // 2 - 1) // 2)
                * ((self.input_shape[1] // 2 // 2 // 2 - 1) // 2)
                * ((self.input_shape[2] // 2 // 2 // 2 - 1) // 2)
                * filters[3]
            )
            out_pad4 = 1 if self.input_shape[0] // 2 // 2 // 2 % 2 == 0 else 0
            self.deconv4 = nn.ConvTranspose3d(
                filters[3],
                filters[2],
                3,
                stride=2,
                padding=0,
                output_padding=out_pad4,
                bias=bias,
            )
            self.bn4 = nn.BatchNorm3d(filters[2])

        if num_layers == 5:
            lin_features_len = (
                ((input_shape[0] // 2 // 2 // 2 // 2 - 1) // 2)
                * ((input_shape[1] // 2 // 2 // 2 // 2 - 1) // 2)
                * ((input_shape[2] // 2 // 2 // 2 // 2 - 1) // 2)
                * filters[4]
            )
            out_pad5 = 1 if input_shape[0] // 2 // 2 // 2 // 2 % 2 == 0 else 0
            self.deconv5 = nn.ConvTranspose3d(
                filters[4],
                filters[3],
                3,
                stride=2,
                padding=0,
                output_padding=out_pad5,
                bias=bias,
            )
            self.bn5 = nn.BatchNorm3d(filters[3])
            out_pad = 1 if input_shape[0] // 2 // 2 // 2 % 2 == 0 else 0
            self.deconv4 = nn.ConvTranspose3d(
                filters[3],
                filters[2],
                5,
                stride=2,
                padding=2,
                output_padding=out_pad,
                bias=bias,
            )
            self.bn4 = nn.BatchNorm3d(filters[2])

        self.bn3 = nn.BatchNorm3d(filters[1])
        self.bn2 = nn.BatchNorm3d(filters[0])

        self.deconv2 = nn.ConvTranspose3d(
            filters[1],
            filters[0],
            5,
            stride=2,
            padding=2,
            output_padding=out_pad2,
            bias=bias,
        )

        self.deconv1 = nn.ConvTranspose3d(
            filters[0],
            input_shape[3],
            5,
            stride=2,
            padding=2,
            output_padding=out_pad1,
            bias=bias,
        )

        self.debedding = nn.Linear(num_features, lin_features_len, bias=bias)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.debedding(x)

        if self.num_layers == 5:
            x = x.view(
                x.size(0),
                self.filters[4],
                ((self.input_shape[0] // 2 // 2 // 2 // 2 - 1) // 2),
                ((self.input_shape[1] // 2 // 2 // 2 // 2 - 1) // 2),
                ((self.input_shape[2] // 2 // 2 // 2 // 2 - 1) // 2),
            )
            x = self.relu5(x)
            x = self.deconv5(x)
            x = self.relu4(x)
            x = self.bn5(x)
            x = self.deconv4(x)
            x = self.relu3(x)
            x = self.bn4(x)

        if self.num_layers == 4:
            x = x.view(
                x.size(0),
                self.filters[3],
                ((self.input_shape[0] // 2 // 2 // 2 - 1) // 2),
                ((self.input_shape[1] // 2 // 2 // 2 - 1) // 2),
                ((self.input_shape[2] // 2 // 2 // 2 - 1) // 2),
            )
            x = self.relu4(x)
            x = self.deconv4(x)
            x = self.relu3(x)
            x = self.bn4(x)

        if self.num_layers == 3:
            x = x.view(
                x.size(0),
                self.filters[2],
                ((self.input_shape[0] // 2 // 2 - 1) // 2),
                ((self.input_shape[1] // 2 // 2 - 1) // 2),
                ((self.input_shape[2] // 2 // 2 - 1) // 2),
            )

        x = self.deconv3(x)
        x = self.relu2(x)
        x = self.bn3(x)
        x = self.deconv2(x)
        x = self.relu1(x)
        x = self.bn2(x)
        x = self.deconv1(x)
        x = self.sig(x)

        return x
