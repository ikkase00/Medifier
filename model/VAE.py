import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Build ResNet blocks
class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, last_layer=False, groups=32):
        super().__init__()

        # Build layers of the block
        self.last_layer = last_layer

        self.conv1gn = nn.GroupNorm(groups, in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.conv2gn = nn.GroupNorm(groups, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        # Add skip connection
        if stride != 1 or in_planes != planes * self.expansion:
            self.cut = nn.Sequential(
                nn.GroupNorm(groups, in_planes),
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
            )
        else:
            self.cut = nn.Identity()

    def forward(self, x):
        out = self.conv1(F.silu(self.conv1gn(x)))
        out = self.conv2(F.silu(self.conv2gn(out)))
        out = out + self.cut(x)
        return out

# The ResNet bottleneck, same concept with the ResBlock but with an additional upsampling
class ResBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, last_layer=False, groups=32):
        super().__init__()

        # Build layers of the model
        self.last_layer = last_layer

        self.conv1gn = nn.GroupNorm(groups, in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)

        self.conv2gn = nn.GroupNorm(groups, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.conv3gn = nn.GroupNorm(groups, planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)

        if stride != 1 or in_planes != planes * self.expansion:
            self.cut = nn.Sequential(
                nn.GroupNorm(groups, in_planes),
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
            )
        else:
            self.cut = nn.Identity()

    def forward(self, x):
        out = self.conv1(F.silu(self.conv1gn(x)))
        out = self.conv2(F.silu(self.conv2gn(out)))
        out = self.conv3(F.silu(self.conv3gn(out)))
        out = out + self.cut(x)
        return out

# Encoder
class VAEEncoder(nn.Module):
    def __init__(self, in_channels, C, r, num_blocks=(2, 2, 2, 2)):
        super().__init__()
        self.C = C # Latent channels
        self.r = r # Timesteps
        self.num_blocks = num_blocks

        # Build layers of the model
        self.stem = nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)

        self.in_planes = 64
        self.layer1 = self._make_layer(ResBlock, self.num_blocks[0], planes=64, stride=1)

        self.down1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.in_planes = 128
        self.layer2 = self._make_layer(ResBlock, self.num_blocks[1], planes=128, stride=1)

        self.down2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.in_planes = 256
        self.bottleneck = self._make_layer(ResBottleneck, self.num_blocks[2], planes=256, stride=1)

        self.mu = nn.Conv2d(256 * ResBottleneck.expansion, C, 1, bias=True)
        self.logvar = nn.Conv2d(256 * ResBottleneck.expansion, C, 1, bias=True)

    def forward(self, x):
        out = self.stem(x)
        out = self.layer1(out)
        out = self.down1(out)
        out = self.layer2(out)
        out = self.down2(out)
        out = self.bottleneck(out)

        # Mean and log variance of the latent space, required by the paper for loss calculation
        mu = self.mu(out)
        logvar = self.logvar(out)

        return mu, logvar

    # helper function to make and stack layers
    def _make_layer(self, block, num_blocks, planes, stride=1):
        layers = []
        layers.append(block(self.in_planes, planes, stride=stride))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, stride=1))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

# Decoder
class VAEDecoder(nn.Module):
    def __init__(
            self,
            out_channels: int,
            C: int,
            num_blocks=(2, 2, 2, 2),
            groups: int = 32,
            target_hw=(80, 512), # Output width and height
            upsample_mode: str = "nearest",
            out_activation: str = "tanh",
            debug_shapes: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.C = C
        self.num_blocks = num_blocks
        self.groups = groups
        self.target_hw = target_hw
        self.upsample_mode = upsample_mode
        self.out_activation = out_activation
        self.debug_shapes = debug_shapes

        self._built = False
        self.proj = None
        self.bottleneck = None
        self.ups = None
        self.head = None

    @staticmethod
    def _is_power_of_two(x: int) -> bool:
        return x > 0 and (x & (x - 1)) == 0

    @staticmethod
    def _g_ok(g: int, c: int) -> int:
        return max(1, math.gcd(g, c))

    def _upsample(self):
        # Choose upsampling technique
        if self.upsample_mode == "bilinear":
            return nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        return nn.Upsample(scale_factor=2, mode=self.upsample_mode)

    # helper function to make and stack layers
    def _make_layer(self, block, in_planes, planes, n_blocks, groups):
        layers = []

        layers.append(block(in_planes, planes, stride=1, groups=groups))
        in_planes = planes * block.expansion
        for _ in range(1, n_blocks):
            layers.append(block(in_planes, planes, stride=1, groups=groups))
            in_planes = planes * block.expansion
        return nn.Sequential(*layers), in_planes

    # Upsample using ResBlocks
    def _upstage(self, in_ch: int, out_ch: int, n_blocks: int) -> nn.Sequential:
        return nn.Sequential(
            nn.GroupNorm(self._g_ok(self.groups, in_ch), in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            *[ResBlock(out_ch, out_ch, groups=self._g_ok(self.groups, out_ch))
              for _ in range(n_blocks)]
        )

    # Figure out how many upsampling steps needed
    def _infer_plan_from_latent(self, h_lat: int, w_lat: int, th: int, tw: int):
        sh, sw = th // h_lat, tw // w_lat
        ups_needed = int(math.log2(sh))

        schedule = [256, 128, 64]

        plan = schedule[-ups_needed:] if ups_needed > 0 else []
        return ups_needed, plan

    # Build decoder base on the latent size z
    def _build_from_z(self, z: torch.Tensor):
        B, Cz, h, w = z.shape
        th, tw = self.target_hw
        ups_needed, plan = self._infer_plan_from_latent(h, w, th, tw)

        base_ch = 512 if ups_needed >= 3 else 256
        exp = getattr(ResBottleneck, "expansion", 1)

        self.proj = nn.Conv2d(Cz, base_ch * exp, kernel_size=1, bias=False)

        n_bottleneck = self.num_blocks[2] if len(self.num_blocks) > 2 else 2

        bottleneck_blocks, cur_ch = self._make_layer(
            ResBottleneck,
            in_planes=base_ch * exp,
            planes=base_ch,
            n_blocks=n_bottleneck,
            groups=self._g_ok(self.groups, base_ch * exp),
        )
        self.bottleneck = bottleneck_blocks

        # Build upsampling layers
        ups = []

        stage_blocks = {
            256: self.num_blocks[2] if len(self.num_blocks) > 2 else 2,
            128: self.num_blocks[1] if len(self.num_blocks) > 1 else 2,
            64: self.num_blocks[0] if len(self.num_blocks) > 0 else 2,
        }

        in_ch = cur_ch
        for out_ch in plan:
            ups.append(self._upsample())
            ups.append(self._upstage(in_ch, out_ch, stage_blocks[out_ch]))
            in_ch = out_ch

        self.ups = nn.Sequential(*ups) if ups else nn.Identity()

        self.head = nn.Sequential(
            nn.GroupNorm(self._g_ok(self.groups, in_ch), in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, self.out_channels, kernel_size=7, padding=3, bias=True),
        )

        self._built = True

    # Apply layers
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if not self._built:
            self._build_from_z(z)

        h = self.proj(z)
        h = self.bottleneck(h)
        h = self.ups(h)
        out = self.head(h)

        # Resize
        th, tw = self.target_hw
        oh, ow = out.shape[-2:]
        if (oh, ow) != (th, tw):
            if max(abs(oh - th), abs(ow - tw)) <= 2:
                out = F.interpolate(out, size=(th, tw), mode="bilinear", align_corners=False)

        # Apply activatiosn
        if self.out_activation == "tanh":
            out = torch.tanh(out)
        elif self.out_activation == "sigmoid":
            out = torch.sigmoid(out)
        return out
