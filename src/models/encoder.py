import torch
import torch.nn as nn
import torch.nn.functional as F

class Chomp1d(nn.Module):
    """Removes the extra padding to keep convolutions causal."""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCNEncoder(nn.Module):
    """
    Encoder that maps a lookback window of features to a latent dimension z_t.
    """
    def __init__(self, input_size, num_channels, latent_dim, kernel_size=2, dropout=0.2):
        super(TCNEncoder, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], latent_dim)

    def forward(self, x):
        """
        x shape: (batch_size, input_size, seq_length)
        """
        y1 = self.tcn(x)
        # Take the output of the last time step
        return self.linear(y1[:, :, -1])

def contrastive_loss(z1, z2, margin=1.0):
    """
    Self-supervised contrastive loss.
    z1 and z2 are embeddings of positive pairs (temporally adjacent windows).
    Negative pairs are created by shifting z2 in the batch.
    """
    # Positive pairs: distance should be minimized
    pos_dist = F.pairwise_distance(z1, z2)
    
    # Negative pairs: created by rolling the batch
    # This assumes batch elements are randomly sampled distant windows,
    # or at least not immediately adjacent except for z1/z2
    z2_neg = torch.roll(z2, shifts=1, dims=0)
    neg_dist = F.pairwise_distance(z1, z2_neg)
    
    loss = torch.mean(pos_dist**2 + torch.clamp(margin - neg_dist, min=0.0)**2)
    return loss
