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
        Constructs a fast 3d convolutional neural network.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        super(Fast3dCNN, self).__init__()

        # XY plane convolution branch
        self.xy_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # XZ plane convolution branch
        self.xz_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # YZ plane convolution branch
        self.yz_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Final fully connected layers
        self.fc1 = nn.Linear(128 * 32 * 32 * 3, 256)
        self.fc2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        Forward pass for the 2.5D CNN.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch_size, 1, height, width, depth).

        Returns
        -------
        torch.Tensor
            Output predictions of shape (batch_size, 1).

        """
        # Unpack slices for each plane
        xy_slices = x[:, :, :, :, x.size(4) // 2]  # Extract XY plane
        xz_slices = x[:, :, :, x.size(3) // 2, :].permute(0, 1, 3, 2)
        yz_slices = x[:, :, x.size(2) // 2, :, :].permute(0, 1, 2, 3)

        # Process each plane with its respective branch
        xy_out = self.xy_branch(xy_slices)
        xz_out = self.xz_branch(xz_slices)
        yz_out = self.yz_branch(yz_slices)

        # Flatten and concatenate outputs
        xy_out = torch.flatten(xy_out, 1)
        xz_out = torch.flatten(xz_out, 1)
        yz_out = torch.flatten(yz_out, 1)
        combined = torch.cat((xy_out, xz_out, yz_out), dim=1)

        # Fully connected layers
        out = self.relu(self.fc1(combined))
        out = self.dropout(out)
        out = self.fc2(out)

        return out
