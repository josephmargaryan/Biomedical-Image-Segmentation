import torch
import torch.nn as nn
from einops import rearrange
import numpy as np
import matplotlib.pyplot as plt

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size=4, embed_dim=96):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        return x

class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=8, attn_dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layernorm = nn.LayerNorm(embed_dim)
        self.multiheadattention = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_dropout, batch_first=True)

    def forward(self, x, return_attention_weights=False):
        x = self.layernorm(x)
        output, attn_weights = self.multiheadattention(query=x, key=x, value=x, need_weights=True)
        if return_attention_weights:
            return output, attn_weights
        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttentionBlock(embed_dim, num_heads, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, return_attention_weights=False):
        if return_attention_weights:
            attn_output, attn_weights = self.attention(self.layernorm1(x), return_attention_weights=True)
            x = x + attn_output
            x = x + self.mlp(self.layernorm2(x))
            return x, attn_weights
        else:
            x = x + self.attention(self.layernorm1(x))
            x = x + self.mlp(self.layernorm2(x))
            return x


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, num_patches):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))

    def forward(self, x):
        return x + self.pos_embed

class SegmentationHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.upsample(x)
        return x


class VisionTransformerSegmentation(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, num_heads, mlp_dim, num_layers, num_classes, image_size):
        super().__init__()
        self.embedding = PatchEmbedding(in_channels, patch_size, embed_dim)
        num_patches = (image_size // patch_size) ** 2  # dynamically calculate num_patches
        self.pos_encoding = PositionalEncoding(embed_dim, num_patches)
        self.encoder = nn.ModuleList([TransformerEncoderLayer(embed_dim, num_heads, mlp_dim) for _ in range(num_layers)])

        self.decoder1 = UpsampleBlock(embed_dim, embed_dim // 2)
        self.decoder2 = UpsampleBlock(embed_dim // 2, embed_dim // 4)
        self.segmentation_head = SegmentationHead(embed_dim // 4, num_classes)

    def forward(self, x, return_attention_weights=False):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        
        attn_weights_list = []
        
        for layer in self.encoder:
            if return_attention_weights:
                x, attn_weights = layer(x, return_attention_weights=True)
                attn_weights_list.append(attn_weights)
            else:
                x = layer(x)

        b, n, c = x.size()
        h = w = int(n ** 0.5)  # assuming square patches
        x = x.permute(0, 2, 1).view(b, c, h, w)  # (b, c, h, w)

        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.segmentation_head(x)
        
        if return_attention_weights:
            return x, attn_weights_list
        else:
            return x




