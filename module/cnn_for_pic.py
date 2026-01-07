import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

# ------------------------- DenseNet backbone -------------------------
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

# ------------------------- Attention blocks -------------------------
class SimpleMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0 
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.w_o(context)
        return output, attention_weights

class LinearAttention(nn.Module):
    def __init__(self, in_features, d_k, d_v, eps=1e-6):
        super().__init__()
        self.q_proj = nn.Linear(in_features, d_k, bias=False)
        self.k_proj = nn.Linear(in_features, d_k, bias=False)
        self.v_proj = nn.Linear(in_features, d_v, bias=False)
        self.eps = eps

    @staticmethod
    def phi(x):
        return F.elu(x) + 1.0

    def forward(self, x):
        B, N, _ = x.shape
        Q = self.q_proj(x)                    
        K = self.k_proj(x)                  
        V = self.v_proj(x)                   

        Qf = self.phi(Q)                      
        Kf = self.phi(K)                     

        S = torch.einsum('bnd,bnv->bdv', Kf, V)
        num = torch.einsum('bnd,bdv->bnv', Qf, S)

        ones = torch.ones(N, device=x.device, dtype=x.dtype)
        z = torch.einsum('bnd,n->bd', Kf, ones)  
        denom = torch.einsum('bnd,bd->bn', Qf, z) + self.eps
        denom = denom.unsqueeze(-1)              

        out = num / denom                  
        return out

# ------------------------- Loss -------------------------
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
# ------------------------- Positional Encoding -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x
# ------------------------- Full model -------------------------
class DenseNetWithLinearAttentionn(nn.Module):
    def __init__(self, num_classes=10, growth=24, bn_size=2, drop_rate=0.2,
                 num_layers_block1=3, num_layers_block2=3, d_k=64, d_v=64, 
                 use_gn_head=False, num_groups=8):
        super().__init__()
        
        self.backbone = SmallDenseNet(
            growth=growth, 
            bn_size=bn_size, 
            drop_rate=drop_rate,
            num_layers_block1=num_layers_block1, 
            num_layers_block2=num_layers_block2
        )
        C = self.backbone.out_ch
        d_k = d_k or C
        d_v = d_v or C

        self.attn = LinearAttention(in_features=C, d_k=d_k, d_v=d_v)
        self.proj = nn.Linear(d_v, C)
        self.ln1 = nn.LayerNorm(C)

        self.ffn = nn.Sequential(
            nn.Linear(C, C * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(C * 2, C)
        )
        self.ln2 = nn.LayerNorm(C)
        
        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(C, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(512, num_classes)
        )

        if use_gn_head:
            self.backbone.head_bn = nn.GroupNorm(num_groups=num_groups, num_channels=self.backbone.out_ch)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Backbone features
        feat = self.backbone(x)                  # (B,C,H,W)
        B, C, H, W = feat.shape
        
        # Reshape for attention
        seq = feat.flatten(2).transpose(1, 2)    # (B, HW, C)
        
        # Attention block với residual
        att_out = self.attn(seq)                 # (B, HW, d_v)
        att_out = self.proj(att_out)             # (B, HW, C)
        seq = self.ln1(seq + att_out)            # residual + norm
        
        # FFN với residual
        ffn_out = self.ffn(seq)
        seq = self.ln2(seq + ffn_out)            # residual + norm
        
        # Global pooling và classification
        pooled = seq.mean(dim=1)                 # (B,C)
        return self.classifier(pooled)
