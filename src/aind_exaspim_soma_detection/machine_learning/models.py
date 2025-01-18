"""
Created on Mon Dec 9 13:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Neural network architectures that classifies 3d image patches with a
soma proposal as accept or reject.

"""

import torch
import torch.nn as nn
import torch.nn.init as init


class Fast3dCNN(nn.Module):
    """
    Fast 3d convolutional neural network that utilizes 2.5d convolutional
    layers to improve the computational complexity.
    """

    def __init__(self, patch_shape):
        """
        Constructs the neural network architecture.

        Parameters
        ----------
        patch_shape : Tuple[int]
            Shape of image patches to be run through network.

        Returns
        -------
        None

        """
        super(Fast3dCNN, self).__init__()
        self.patch_shape = patch_shape

        # Convolutional layer 1
        self.layer1 = nn.Sequential(
            FastConvLayer(1, 16),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Dropout3d(0.25),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        # Convolutional layer 2
        self.layer2 = nn.Sequential(
            FastConvLayer(16, 32),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Dropout3d(0.25),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        # Convolutional layer 3
        self.layer3 = nn.Sequential(
            FastConvLayer(32, 64),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Dropout3d(0.25),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        # Convolutional layer 4
        self.layer4 = nn.Sequential(
            FastConvLayer(64, 128),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Dropout3d(0.25),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        # Final fully connected layers
        self.output = nn.Sequential(
            nn.Linear(128 * (self.patch_shape[0] // 16) ** 3, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
        )

        # Initialize weights
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass of the 2.5D convolutional neural network.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch_size, in_channels, height, width, depth).

        Returns
        -------
        torch.Tensor
            Output with shape (batch_size, 1).

        """
        # Convolutional Layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Output layer
        x = torch.flatten(x, start_dim=1)
        x = self.output(x)
        return x


class FastConvLayer(nn.Module):
    """
    Class that performs a single layer of 2.5 convolution.

    """

    def __init__(self, in_channels, out_channels):
        """
        Constructs a 2.5D convolutional layer.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        int_channels : int
            Number of intermediate channels for the 2D convolutions.

        Returns
        -------
        None

        """
        super(FastConvLayer, self).__init__()
        self.conv_2d = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv_3d = nn.Conv3d(3 * in_channels, out_channels, 3, padding=1)

    def forward(self, x):
        """
        Forward pass of a single 2.5D convolution block.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch_size, in_channels, height, width, depth).

        Returns
        -------
        torch.Tensor
            Output of shape (batch_size, out_channels, height, width, depth).

        """
        B, C, D, H, W = x.shape

        # Process XY slices
        xy_slices = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
        xy_features = self.conv_2d(xy_slices).reshape(B, D, -1, H, W)
        xy_features = xy_features.permute(0, 2, 1, 3, 4)

        # Process XZ slices
        xz_slices = x.permute(0, 3, 1, 2, 4).reshape(B * H, C, D, W)
        xz_features = self.conv_2d(xz_slices).reshape(B, H, -1, D, W)
        xz_features = xz_features.permute(0, 2, 3, 1, 4)

        # Process YZ slices
        yz_slices = x.permute(0, 4, 1, 2, 3).reshape(B * W, C, D, H)
        yz_features = self.conv_2d(yz_slices).reshape(B, W, -1, D, H)
        yz_features = yz_features.permute(0, 2, 3, 4, 1)

        # Fuse features using 3D convolution
        combined = torch.cat([xy_features, xz_features, yz_features], dim=1)
        output = self.conv_3d(combined)
        return output
