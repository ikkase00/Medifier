import torch
import torch.nn as nn
import math

class SinusoidoPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):

        device = time.device
        half_dim = self.dim // 2

        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device = device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim = -1)

        return embeddings

class UBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up = False):
        super().__init__()

        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding = 1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding = 1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding = 1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t, ):
        a = self.bn(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]

        a = a + time_emb
        a = self.bn(self.relu(self.conv2(a)))
        return self.transform(a)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.image_ch = 1
        out_ch = 1
        time_emb_dim = 32
        down_chs = (64, 128, 256, 512)
        up_chs = (512, 256, 128, 64)

        self.time_mlp = nn.Sequential(
            SinusoidoPositionalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )

        self.init = nn.Conv2d(self.image_ch, down_chs[0], 3, padding = 1)
        self.down = nn.ModuleList([UBlock(down_chs[i], down_chs[i + 1], time_emb_dim) for i in range(len(down_chs) - 1)])
        self.up = nn.ModuleList([UBlock(up_chs[i], up_chs[i + 1], time_emb_dim, True) for i in range(len(up_chs) - 1)])
        self.out = nn.Conv2d(up_chs[-1], out_ch, 1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x = self.init(x)
        resids = []

        for d in self.down:
            x = d(x, t)
            resids.append(x)

        for u in self.up:
            resid = resids.pop()
            x = torch.cat((x, resid), dim = 1)
            x = u(x, t)

        return self.out(x)