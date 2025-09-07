import torch
import torch.nn as nn
import math

# Positional embedding from the DDPM paper
class SinusoidoPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2

        # Compute frequency scales
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)

        # Multiply and concatnate time values
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        return embeddings


# UNet building block
class UBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()

        # Linear layer to inject time embeddings
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        if up:
            # Upsampling block
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            # Downsampling block
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        # First layer (convolution + normalization + activation)
        a = self.bn(self.relu(self.conv1(x)))

        # Add time embedding
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(...,) + (None,) * 2]
        a = a + time_emb

        # Second layer (convolution + normalization + activation)
        a = self.bn(self.relu(self.conv2(a)))

        # Transform (downsample or upsample)
        return self.transform(a)


# UNet model
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.image_ch = 1   # Input channel, 1 because mel spectrogram is grayscale
        out_ch = 1          # Output channel
        time_emb_dim = 32   # Dimension of time embeddings

        # Channel sizes for downsampling and upsampling
        down_chs = (64, 128, 256, 512)
        up_chs = (512, 256, 128, 64)

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidoPositionalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )

        # Initial convolution
        self.init = nn.Conv2d(self.image_ch, down_chs[0], 3, padding=1)

        # Downsampling blocks
        self.down = nn.ModuleList([
            UBlock(down_chs[i], down_chs[i + 1], time_emb_dim)
            for i in range(len(down_chs) - 1)
        ])

        # Upsampling blocks
        self.up = nn.ModuleList([
            UBlock(up_chs[i], up_chs[i + 1], time_emb_dim, True)
            for i in range(len(up_chs) - 1)
        ])

        # Output layer
        self.out = nn.Conv2d(up_chs[-1], out_ch, 1)

    def forward(self, x, timestep):
        # Get time embeddings
        t = self.time_mlp(timestep)

        x = self.init(x)
        resids = []

        # Downsampling
        for d in self.down:
            x = d(x, t)
            resids.append(x)

        # Upsampling with skip connection
        for u in self.up:
            resid = resids.pop()   # Get corresponding skip connection
            x = torch.cat((x, resid), dim=1)  # Concatenate along channel axis
            x = u(x, t)

        # Output layer 
        return self.out(x)
