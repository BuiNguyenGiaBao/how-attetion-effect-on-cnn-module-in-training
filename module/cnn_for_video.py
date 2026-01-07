import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

#FRAME BACKBONE (CNN)
class DenseLayer(nn.Module):
    def __init__(self, num_inputs, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_inputs)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_inputs, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)

        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = self.conv1(self.relu1(self.norm1(x)))
        new_features = self.conv2(self.relu2(self.norm2(new_features)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], dim=1)
    
class DenseBlock(nn.Module):
    def __init__(self, num_inputs, growth_rate, bn_size, drop_rate, num_layers):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_ch = num_inputs + i * growth_rate
            layers.append(DenseLayer(in_ch, growth_rate, bn_size, drop_rate))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Transition(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.norm = nn.BatchNorm2d(num_inputs)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_inputs, num_outputs, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(self.relu(self.norm(x))))

class SmallDenseNet(nn.Module):
    def __init__(self, growth=32, bn_size=4, drop_rate=0.0, num_layers_block1=4, num_layers_block2=4):
        super().__init__()
        init_ch = 2 * growth
        self.stem = nn.Sequential(
            nn.Conv2d(3, init_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(init_ch),
            nn.ReLU(inplace=True),
        )
        # Block 1
        self.block1 = DenseBlock(init_ch, growth, bn_size, drop_rate, num_layers_block1)
        ch1 = init_ch + num_layers_block1 * growth
        self.trans1 = Transition(ch1, ch1 // 2)
        # Block 2
        ch_in2 = ch1 // 2
        self.block2 = DenseBlock(ch_in2, growth, bn_size, drop_rate, num_layers_block2)
        ch2 = ch_in2 + num_layers_block2 * growth
        # Head
        self.head_bn = nn.BatchNorm2d(ch2)
        self.head_relu = nn.ReLU(inplace=True)
        self.out_ch = ch2

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.trans1(x)
        x = self.block2(x)
        x = self.head_relu(self.head_bn(x))
        return x


#TCN (Temporal Convolutional Network)

def _causal_padding(kernel_size: int, dilation: int) -> int:
    return (kernel_size - 1) * dilation

class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
        use_weight_norm: bool = True,
        causal: bool = True,
    ):
        super().__init__()
        pad = _causal_padding(kernel_size, dilation) if causal else (kernel_size // 2) * dilation

        conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)

        if use_weight_norm:
            from torch.nn.utils import weight_norm
            conv1 = weight_norm(conv1)
            conv2 = weight_norm(conv2)

        self.conv1 = conv1
        self.conv2 = conv2
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        self.causal = causal
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

        for m in [self.conv1, self.conv2] + ([self.downsample] if self.downsample is not None else []):
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        out = self.conv1(x)
        if self.causal:
            out = out[..., :x.size(-1)]
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        if self.causal:
            out = out[..., :x.size(-1)]
        out = self.relu(out)
        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(
        self,
        in_ch: int,
        channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.1,
        use_weight_norm: bool = True,
        causal: bool = True,
    ):
        super().__init__()
        # Accept both int and list/tuple for channels
        # (Older code sometimes calls TemporalConvNet(in_ch, channels=int))
        if isinstance(channels, int):
            channels = [channels]
        elif isinstance(channels, tuple):
            channels = list(channels)
        elif channels is None:
            raise ValueError('channels must be an int or a list/tuple of ints')

        layers = []
        cur_in = in_ch
        for i, ch in enumerate(channels):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    in_ch=cur_in,
                    out_ch=ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    use_weight_norm=use_weight_norm,
                    causal=causal,
                )
            )
            cur_in = ch
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        return self.network(x)

    @property
    def receptive_field(self) -> int:
        rf = 1
        for m in self.network:
            k = m.conv1.kernel_size[0]
            d = m.conv1.dilation[0]
            rf += 2 * (k - 1) * d
        return rf

#GRU TEMPORAL ENCODER
class GRUEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bidirectional: bool = False, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_dim = hidden_size * (2 if bidirectional else 1)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, C)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out_packed, h = self.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        else:
            out, h = self.gru(x)
        # Use last hidden state (concatenate directions if bidirectional)
        if isinstance(h, torch.Tensor):
            last = torch.cat([h[-2], h[-1]], dim=-1) if self.gru.bidirectional else h[-1]
        else:
            last = out[:, -1, :]
        return last  # (B, out_dim)

#VIDEO CLASSIFIER WITH TCN / GRU
class VideoClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        growth: int = 24,
        bn_size: int = 2,
        drop_rate: float = 0.2,
        num_layers_block1: int = 3,
        num_layers_block2: int = 3,
        temporal_mode: str = "tcn",         # "tcn", "gru", "tcn_gru", "avg"
        tcn_channels: List[int] = None,     # e.g., [128, 128, 256]
        tcn_kernel_size: int = 3,
        tcn_dropout: float = 0.1,
        gru_hidden: int = 256,
        gru_layers: int = 1,
        gru_bidirectional: bool = False,
        temporal_pool: str = "avg",         # for TCN: "avg" | "last" | "max"
    ):
        super().__init__()
        self.temporal_mode = temporal_mode
        self.temporal_pool = temporal_pool

        # Frame encoder
        self.backbone = SmallDenseNet(
            growth=growth, 
            bn_size=bn_size, 
            drop_rate=drop_rate,
            num_layers_block1=num_layers_block1, 
            num_layers_block2=num_layers_block2
        )
        C = self.backbone.out_ch

        # Temporal modules
        if temporal_mode in ("tcn", "tcn_gru"):
            if tcn_channels is None:
                tcn_channels = [128, 128, 256]
            self.tcn = TemporalConvNet(in_ch=C, channels=tcn_channels, kernel_size=tcn_kernel_size, dropout=tcn_dropout, causal=True)
            self.tcn_out = tcn_channels[-1]
        else:
            self.tcn = None
            self.tcn_out = C

        if temporal_mode in ("gru", "tcn_gru"):
            self.gru = GRUEncoder(input_size=(self.tcn_out if temporal_mode == "tcn_gru" else C),
                                  hidden_size=gru_hidden, num_layers=gru_layers, bidirectional=gru_bidirectional, dropout=drop_rate)
            self.gru_out = self.gru.out_dim
        else:
            self.gru = None
            self.gru_out = self.tcn_out if temporal_mode == "tcn" else C

        # Classifier head
        head_in = {
            "avg": C,
            "tcn": self.tcn_out,
            "gru": self.gru_out,
            "tcn_gru": self.gru_out,
        }[temporal_mode]
        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(head_in, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(512, num_classes),
        )

    def _encode_frames(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 3, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        feat = self.backbone(x)            # (B*T, C_f, H', W')
        feat = F.adaptive_avg_pool2d(feat, output_size=1).squeeze(-1).squeeze(-1)  # (B*T, C_f)
        feat = feat.view(B, T, -1)         # (B, T, C_f)
        return feat

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Encode frames
        seq = self._encode_frames(x)       # (B, T, C_f)

        if self.temporal_mode == "avg":
            pooled = seq.mean(dim=1)       # (B, C_f)
            return self.classifier(pooled)

        if self.temporal_mode in ("tcn", "tcn_gru"):
            # TCN expects (B, C, T)
            tcn_in = seq.transpose(1, 2)   # (B, C_f, T)
            tcn_feat = self.tcn(tcn_in)    # (B, C_out, T)
            if self.temporal_mode == "tcn":
                if self.temporal_pool == "avg":
                    pooled = tcn_feat.mean(dim=-1)
                elif self.temporal_pool == "max":
                    pooled = tcn_feat.amax(dim=-1)
                elif self.temporal_pool == "last":
                    pooled = tcn_feat[..., -1]
                else:
                    raise ValueError("temporal_pool must be 'avg' | 'max' | 'last'")
                return self.classifier(pooled)
            else:
                # pass to GRU
                seq = tcn_feat.transpose(1, 2)  # (B, T, C_out)

        if self.temporal_mode == "gru":
            # seq already (B, T, C_f)
            pass

        # GRU path
        last = self.gru(seq, lengths=lengths)    # (B, gru_out)
        return self.classifier(last)

# FOCAL LOSS
class FocalLoss(nn.Module):
    def __init__(self, class_weights: Optional[torch.Tensor]=None, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction
        self.register_buffer("class_weights", None)
        if class_weights is not None:
            w = torch.as_tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", w)

    def forward(self, inputs, targets):
        weight = self.class_weights
        ce = F.cross_entropy(inputs, targets, weight=weight, reduction='none')
        pt = torch.exp(-ce)
        loss = self.alpha * (1 - pt)**self.gamma * ce
        if self.reduction == 'mean': return loss.mean()
        if self.reduction == 'sum':  return loss.sum()
        return loss


#MODULE

class _TemporalEncoder(nn.Module):

    def __init__(self, d_model: int, mode: str = "tcn", tcn_kernel: int = 3, tcn_layers: int = 2, tcn_dropout: float = 0.1,
                 gru_layers: int = 1, bidirectional: bool = False, dropout: float = 0.0):
        super().__init__()
        self.mode = mode
        self.d_model = d_model
        if mode == "tcn":
            # stack TCN layers keeping channel = d_model
            chans = [d_model] * tcn_layers
            self.tcn = TemporalConvNet(in_ch=d_model, channels=chans, kernel_size=tcn_kernel, dropout=tcn_dropout, causal=True)
        elif mode == "gru":
            self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=gru_layers,
                               batch_first=True, bidirectional=bidirectional, dropout=dropout if gru_layers > 1 else 0.0)
            # FIX: Only use projection if bidirectional, otherwise use identity
            if bidirectional:
                self.proj = nn.Linear(d_model * 2, d_model)
            else:
                self.proj = nn.Identity()
        else:
            raise ValueError("mode phải là 'tcn' hoặc 'gru'")

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: (B, T, D)
        if self.mode == "tcn":
            x = seq.transpose(1, 2)            # (B, D, T)
            y = self.tcn(x).transpose(1, 2)    # (B, T, D)
            return y
        else:
            y, _ = self.gru(seq)                # (B, T, D*) where D* = D or 2D
            y = self.proj(y)                    # FIX: Always apply proj (could be identity)
            return y

class DenseNetWithTemporalResidual(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        growth: int = 24,
        bn_size: int = 2,
        drop_rate: float = 0.2,
        num_layers_block1: int = 3,
        num_layers_block2: int = 3,
        # temporal
        temporal_mode: str = "tcn",       # "tcn" | "gru"
        d_model: int = None,             
        tcn_kernel: int = 3,
        tcn_layers: int = 2,
        tcn_dropout: float = 0.1,
        gru_layers: int = 1,
        gru_bidirectional: bool = False,
        temporal_pool: str = "avg",       # "avg" | "max" | "last"
        mlp_hidden: int = 512,
        use_gn_head: bool = False,
        num_groups: int = 8,
    ):
        super().__init__()
        self.backbone = SmallDenseNet(
            growth=growth, bn_size=bn_size, drop_rate=drop_rate,
            num_layers_block1=num_layers_block1, num_layers_block2=num_layers_block2
        )
        C = self.backbone.out_ch
        D = d_model or C

        # FIX: Replace head_bn before forward if use_gn_head is True
        if use_gn_head:
            self.backbone.head_bn = nn.GroupNorm(num_groups=num_groups, num_channels=self.backbone.out_ch)
        self.in_proj = nn.Linear(C, D) if D != C else nn.Identity()
        self.temporal = _TemporalEncoder(
            d_model=D, mode=temporal_mode, tcn_kernel=tcn_kernel, tcn_layers=tcn_layers, tcn_dropout=tcn_dropout,
            gru_layers=gru_layers, bidirectional=gru_bidirectional, dropout=drop_rate
        )
        self.ln1 = nn.LayerNorm(D)
        self.ffn = nn.Sequential(
            nn.Linear(D, D * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(D * 2, D),
        )
        self.ln2 = nn.LayerNorm(D)

        self.temporal_pool = temporal_pool
        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(D, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(mlp_hidden, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def _frame_encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 3, H, W) -> (B, T, C)
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        feat = self.backbone(x)                  # (B*T, Cb, H', W')
        feat = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)  # (B*T, Cb)
        seq = feat.view(B, T, -1)               # (B, T, Cb)
        seq = self.in_proj(seq)                 # (B, T, D)
        return seq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = self._frame_encode(x)             # (B, T, D)

        # Temporal block + residual + LN
        y = self.temporal(seq)                  # (B, T, D)
        seq = self.ln1(seq + y)

        # FFN + residual + LN
        y2 = self.ffn(seq)
        seq = self.ln2(seq + y2)

        # Pool theo thời gian
        if self.temporal_pool == "avg":
            pooled = seq.mean(dim=1)
        elif self.temporal_pool == "max":
            pooled = seq.amax(dim=1)
        elif self.temporal_pool == "last":
            pooled = seq[:, -1, :]
        else:
            raise ValueError("temporal_pool must be 'avg' | 'max' | 'last'")

        return self.classifier(pooled)


