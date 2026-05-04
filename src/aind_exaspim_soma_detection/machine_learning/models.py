"""
Created on Mon Dec 9 13:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Neural network architectures that classifies 3d image patches with a
soma proposal as accept or reject.

"""

import torch
import torch.nn as nn


class CNN3D(nn.Module):
    """
    Class that implements a convolutional neural network for 3D images.
    """

    def __init__(
        self,
        patch_shape,
        output_dim=1,
        dropout=0.1,
        n_conv_layers=5,
        n_feat_channels=16,
        use_double_conv=True,
    ):
        """
        Instantiates a CNN3D object.

        Parameters
        ----------
        patch_shape : Tuple[int]
            Shape of input image patch.
        output_dim : int, optional
            Dimension of output. Default is 1.
        dropout : float, optional
            Fraction of values to randomly drop during training. Default is
            0.1.
        n_conv_layers : int, optional
            Number of convolutional layers. Default is 5.
        use_double_conv : bool, optional
            Indication of whether to use double convolution. Default is True.
        """
        # Call parent class
        nn.Module.__init__(self)

        # Class attributes
        self.dropout = dropout
        self.patch_shape = patch_shape

        # Convolutional layers
        self.conv_layers = init_cnn3d(
            1, n_feat_channels, n_conv_layers, use_double_conv=use_double_conv
        )

        # Output layer
        flat_size = self._get_flattened_size()
        self.output = FeedForwardNet(flat_size, output_dim, 3)

        # Initialize weights
        self.apply(self.init_weights)

    def _get_flattened_size(self):
        """
        Compute the flattened feature vector size after applying a sequence
        of convolutional and pooling layers on an input tensor with the given
        shape.

        Returns
        -------
        int
            Length of the flattened feature vector after the convolutions and
            pooling.
        """
        with torch.no_grad():
            x = torch.zeros(1, 1, *self.patch_shape)
            x = self.conv_layers(x)
            return x.view(1, -1).size(1)

    @staticmethod
    def init_weights(m):
        """
        Initializes the weights and biases of a given PyTorch layer.

        Parameters
        ----------
        m : nn.Module
            PyTorch layer or module.
        """
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(
                m.weight, mode="fan_in", nonlinearity="leaky_relu"
            )
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Passes the given input through this neural network.

        Parameters
        ----------
        x : torch.Tensor
            Input vector of features.

        Returns
        -------
        x : torch.Tensor
            Output of the neural network.
        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


# --- Build Simple Neural Networks ---
def init_cnn3d(in_channels, n_feat_channels, n_layers, use_double_conv=True):
    """
    Initializes a convolutional neural network.

    Parameters
    ----------
    in_channels : int
        Number of channels that are input to this convolutional layer.
    out_channels : int
        Number of channels that are output from this convolutional layer.
    n_layers : int
        Number of layers in the network.
    use_double_conv : bool, optional
        Indication of whether to use double convolution. Default is True.

    Returns
    -------
    layers : torch.nn.Sequential
        Sequence of operations that define the network.
    """
    layers = list()
    in_channels = in_channels
    out_channels = n_feat_channels
    for i in range(n_layers):
        # Build layer
        layers.append(
            init_conv_layer(in_channels, out_channels, 3, use_double_conv)
        )

        # Update channel sizes
        in_channels = out_channels
        out_channels *= 2
    return nn.Sequential(*layers)


def init_conv_layer(in_channels, out_channels, kernel_size, use_double_conv):
    """
    Initializes a convolutional layer.

    Parameters
    ----------
    in_channels : int
        Number of channels that are input to this convolutional layer.
    out_channels : int
        Number of channels that are output from this convolutional layer.
    kernel_size : int
        Size of kernel used on convolutional layers.
    use_double_conv : bool
        Indication of whether to use double convolution.

    Returns
    -------
    layers : torch.nn.Sequential
        Sequence of operations that define this convolutional layer.
    """
    # Convolution
    layers = [
        nn.Conv3d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        ),
        nn.BatchNorm3d(out_channels),
        nn.GELU(),
    ]
    if use_double_conv:
        layers += [
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
        ]
    # Pooling
    layers.append(nn.MaxPool3d(kernel_size=2))
    return nn.Sequential(*layers)


class FeedForwardNet(nn.Module):
    """
    A class that implements a feed forward neural network.
    """

    def __init__(self, input_dim, output_dim, n_layers):
        """
        Instantiates a FeedFowardNet object.

        Parameters
        ----------
        input_dim : int
            Dimension of the input.
        output_dim : int
            Dimension of the output of the network.
        n_layers : int
            Number of layers in the network.
        """
        # Call parent class
        super().__init__()

        # Instance attributes
        assert n_layers > 1
        self.net = self.build_network(input_dim, output_dim, n_layers)

    def build_network(self, input_dim, output_dim, n_layers):
        # Set input/output dimensions
        input_dim_i = input_dim
        output_dim_i = max(input_dim // 2, 4)

        # Build architecture
        layers = []
        for i in range(n_layers):
            mlp = init_mlp(input_dim_i, input_dim_i * 2, output_dim_i)
            layers.append(mlp)

            input_dim_i = output_dim_i
            output_dim_i = (
                max(output_dim_i // 2, 4) if i < n_layers - 2 else output_dim
            )

        # Initialize weights
        net = nn.Sequential(*layers)
        net.apply(self._init_weights)
        return net

    @staticmethod
    def _init_weights(m):
        """
        Initializes weights for linear layers using Kaiming initialization.

        Parameters
        ----------
        m : torch.nn.Module
            Module to initialize.
        """
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Passes the given input through this neural network.

        Parameters
        ----------
        x : torch.Tensor
            Input vector of features.

        Returns
        -------
        x : torch.Tensor
            Output of the neural network.
        """
        return self.net(x)


def init_mlp(input_dim, hidden_dim, output_dim, dropout=0.1):
    """
    Initializes a multi-layer perceptron (MLP).

    Parameters
    ----------
    input_dim : int
        Dimension of input.
    hidden_dim : int
        Dimension of the hidden layer.
    output_dim : int
        Dimension of output.
    dropout : float, optional
        Fraction of values to randomly drop during training. Default is 0.1.

    Returns
    -------
    mlp : nn.Sequential
        Multi-layer perception network.
    """
    mlp = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LeakyReLU(),
        nn.Dropout(p=dropout),
        nn.Linear(hidden_dim, output_dim),
    )
    return mlp
