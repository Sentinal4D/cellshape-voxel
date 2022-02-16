import torch.nn as nn
import copy


class ConvolutionalEncoder(nn.Module):
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
        super(ConvolutionalEncoder, self).__init__()
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

        self.conv1 = nn.Conv3d(
            1, filters[0], 5, stride=2, padding=2, bias=bias
        )
        self.bn1 = nn.BatchNorm3d(filters[0])
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=neg_slope)
        else:
            self.relu = nn.ReLU(inplace=False)

        self.relu1 = copy.deepcopy(self.relu)
        self.conv2 = nn.Conv3d(
            filters[0], filters[1], 5, stride=2, padding=2, bias=bias
        )
        self.bn2 = nn.BatchNorm3d(filters[1])
        self.relu2 = copy.deepcopy(self.relu)
        self.relu3 = copy.deepcopy(self.relu)
        if num_layers == 3:
            self.conv3 = nn.Conv3d(
                filters[1], filters[2], 3, stride=2, padding=0, bias=bias
            )
            lin_features_len = (
                ((self.input_shape[0] // 2 // 2 - 1) // 2)
                * ((self.input_shape[1] // 2 // 2 - 1) // 2)
                * ((self.input_shape[2] // 2 // 2 - 1) // 2)
                * filters[2]
            )
            self.embedding = nn.Linear(
                lin_features_len, num_features, bias=bias
            )
        else:
            self.bn2 = nn.BatchNorm3d(filters[1])
            self.conv3 = nn.Conv3d(
                filters[1], filters[2], 5, stride=2, padding=2, bias=bias
            )
        if num_layers == 4:
            self.bn3 = nn.BatchNorm3d(filters[2])
            self.conv4 = nn.Conv3d(
                filters[2], filters[3], 3, stride=2, padding=0, bias=bias
            )
            self.relu4 = copy.deepcopy(self.relu)
            lin_features_len = (
                ((self.input_shape[0] // 2 // 2 // 2 - 1) // 2)
                * ((self.input_shape[1] // 2 // 2 // 2 - 1) // 2)
                * ((self.input_shape[2] // 2 // 2 // 2 - 1) // 2)
                * filters[3]
            )
            self.embedding = nn.Linear(
                lin_features_len, num_features, bias=bias
            )
        if num_layers == 5:
            self.bn3 = nn.BatchNorm3d(filters[2])
            self.conv4 = nn.Conv3d(
                filters[2], filters[3], 5, stride=2, padding=2, bias=bias
            )
            self.relu4 = copy.deepcopy(self.relu)
            self.bn4 = nn.BatchNorm3d(filters[3])
            self.conv5 = nn.Conv3d(
                filters[3], filters[4], 3, stride=2, padding=0, bias=bias
            )
            self.relu5 = copy.deepcopy(self.relu)
            lin_features_len = (
                ((input_shape[0] // 2 // 2 // 2 // 2 - 1) // 2)
                * ((input_shape[1] // 2 // 2 // 2 // 2 - 1) // 2)
                * ((input_shape[2] // 2 // 2 // 2 // 2 - 1) // 2)
                * filters[4]
            )
            self.embedding = nn.Linear(
                lin_features_len, num_features, bias=bias
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.relu3(x)

        if self.num_layers == 4:
            x = self.bn3(x)
            x = self.conv4(x)
            x = self.relu4(x)

        if self.num_layers == 5:
            x = self.bn3(x)
            x = self.conv4(x)
            x = self.relu4(x)
            x = self.bn4(x)
            x = self.conv5(x)
            x = self.relu5(x)

        x = x.view(x.size(0), -1)
        features = self.embedding(x)

        return features
