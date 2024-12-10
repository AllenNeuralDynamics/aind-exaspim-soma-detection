"""
Created on Mon Dec 9 13:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Neural network architectures that classifies 3d image patches with a
soma proposal as accept or reject.

"""

import torch
import torch.nn as nn


class Fast3dCNN(nn.Module):
    def __init__(self):
        """
        Constructs a fast 3d convolutional neural network that utilizes 2.5d
        convolutional layers to improve the computational complexity.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        super(Fast3dCNN, self).__init__()

        # Convolutional Layer 1
        self.conv1 = nn.Sequential(
            FastConvLayer(1, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Convolutional Layer 2
        self.conv1 = nn.Sequential(
            FastConvLayer(32, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Convolutional Layer 3
        self.conv1 = nn.Sequential(
            FastConvLayer(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Final fully connected layers
        self.output = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

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
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Output layer
        x = self.output(x)
        return output


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

        # Convolutions for each plane
        self.conv_xy = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv_xz = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv_yz = nn.Conv2d(in_channels, in_channels, 3, padding=1)

        # Fusion layer to combine features
        self.conv_xyz = nn.Conv3d(3 * in_channels, out_channels, 3, padding=1)

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
        xy_features = self.conv_xy(xy_slices)
        xy_features = xy_features.reshape(B, D, -1, H, W).permute(
            0, 2, 1, 3, 4
        )

        # Process XZ slices
        xz_slices = x.permute(0, 3, 1, 2, 4).reshape(B * H, C, D, W)
        xz_features = self.conv_xz(xz_slices)
        xz_features = xz_features.reshape(B, H, -1, D, W).permute(
            0, 2, 3, 1, 4
        )

        # Process YZ slices
        yz_slices = x.permute(0, 4, 1, 2, 3).reshape(B * W, C, D, H)
        yz_features = self.conv_yz(yz_slices)
        yz_features = yz_features.reshape(B, W, -1, D, H).permute(
            0, 2, 3, 4, 1
        )

        # Fuse features using 3D convolution
        combined = torch.cat([xy_features, xz_features, yz_features], dim=1)
        output = self.conv_xyz(combined)
        return output
