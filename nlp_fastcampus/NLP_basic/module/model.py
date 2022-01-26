import torch
import torch.nn as nn

class Block(nn.Module):

    def __init__(self,
                input_size,
                output_size, 
                use_batch_norm=True, 
                dropout_p=.4):
        self.input_size = input_size
        self.output_size = output_size
        self.use_batch_norm = use_batch_norm
        self.dropdout_p = dropout_p

        super().__init__()

        def get_regularizer(use_batch_norm, size):
            return nn.BatchNorm1d(size) if use_batch_norm else nn.Dropout(dropout_p)

        self.block = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.LeakyReLU(),
            get_regularizer(use_batch_norm, output_size)
        )

    def forward(self, x):
        # |x| = (batch_size, input_size)
        y = self.block(x)
        # |y| = (batch_size, output_size)
        return y


class ImageClassifier(nn.Module):

    def __init__(self,
                input_size,
                output_size,
                hidden_sizes=[500, 400, 300, 200, 100],
                use_batch_norm=True,
                dropout_p=.3):

        super().__init__()
        assert len(hidden_sizes) > 0, 'You need to specifiy hidden layers'

        last_hidden_size = input_size
        blocks = []
        for hidden_size in hidden_sizes:
            blocks += [Block(
                last_hidden_size,
                hidden_size,
                use_batch_norm,
                dropout_p
            )]
            last_hidden_size = hidden_size

        self.layers = nn.Sequential(
            *blocks,
            nn.Linear(last_hidden_size, output_size),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        y = self.layers(x)

        return y


class Autoencoder(nn.Module):

                      # bottleneck size
    def __init__(self, btl_size=2):
        self.btl_size = btl_size

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.BatchNorm1d(200),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.BatchNorm1d(20),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.BatchNorm1d(10),
            nn.Linear(10, btl_size),
        )

        self.decoder = nn.Sequential(
            nn.Linear(btl_size, 10),
            nn.ReLU(),
            nn.BatchNorm1d(10),
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.BatchNorm1d(20),
            nn.Linear(20, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.BatchNorm1d(200),
            nn.Linear(200, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
            nn.Linear(500, 28 * 28),
        )

    def forward(self, x):
        # |x| = (batch_size, 28*28)
        # |z| = (batch_size, btl_size)
        z = self.encoder(x)
        y = self.decoder(z)
        # |y| = |x|

        return y