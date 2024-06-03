import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
import matplotlib.pyplot as plt


class Embedding(nn.Module):
    '''
    input shape -> (b, c, h, w)
    output shape -> (b, num_patches, embed_dim)
    '''

    def __init__(self, in_channels, patch_size=4, embed_dim=96):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.linear_embedding = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.linear_embedding(x)
        x = rearrange(x, 'b c (h1) (w1) -> b (h1 w1) c', h1=h//self.patch_size, w1=w//self.patch_size)
        x = self.relu(self.layer_norm(x))
        return x


class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, embedding_dims, num_heads=4, attn_dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layernorm = nn.LayerNorm([embedding_dims])  # Adjusted normalized_shape
        self.multiheadattention = nn.MultiheadAttention(num_heads=num_heads, embed_dim=embedding_dims, dropout=attn_dropout, batch_first=True)

    def forward(self, x, return_attention_map=False):
        x = self.layernorm(x)
        output, attention_weights = self.multiheadattention(query=x, key=x, value=x, need_weights=True, average_attn_weights=True)
        if return_attention_map:
            return output, attention_weights
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, num_patches):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))

    def forward(self, x):
        batch_size, num_patches, embed_dim = x.size()
        pos_embed = self.pos_embed[:, :num_patches, :]
        return x + pos_embed


class ClassificationToken(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x):
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        return torch.cat((cls_tokens, x), dim=1)


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
            attn_output, attn_weights = self.attention(self.layernorm1(x), return_attention_map=True)
            x = x + attn_output
            x = x + self.mlp(self.layernorm2(x))
            return x, attn_weights
        else:
            x = x + self.attention(self.layernorm1(x))
            x = x + self.mlp(self.layernorm2(x))
            return x


class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        cls_token = x[:, 0]  # The first token is the [CLS] token
        return self.fc(cls_token)


class VisionTransformer(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, num_heads, mlp_dim, num_layers, num_classes):
        super().__init__()
        self.embedding = Embedding(in_channels, patch_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, (256 // patch_size) ** 2)
        self.cls_token = ClassificationToken(embed_dim)
        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_dim)
            for _ in range(num_layers)
        ])
        self.classification_head = ClassificationHead(embed_dim, num_classes)

    def forward(self, x, return_attention_weights=False):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.cls_token(x)
        
        attn_weights_list = []

        for layer in self.encoder:
            if return_attention_weights:
                x, attn_weights = layer(x, return_attention_weights=True)
                attn_weights_list.append(attn_weights)
            else:
                x = layer(x)

        x = self.classification_head(x)

        if return_attention_weights:
            return x, attn_weights_list
        else:
            return x


def plot_attention_map(attention, image, patch_size, layer, head):
    num_patches = int(np.sqrt(attention.shape[-1]))
    attention = attention.reshape(num_patches, num_patches)
    attention_resized = np.kron(attention, np.ones((patch_size, patch_size)))

    plt.imshow(image, cmap='gray')
    plt.imshow(attention_resized, cmap='jet', alpha=0.5)  # overlay attention map
    plt.title(f"Layer {layer}, Head {head}")
    plt.colorbar()
    plt.show()


def test_vision_transformer():
    # Define the parameters
    in_channels = 3
    patch_size = 4
    embed_dim = 96
    num_heads = 3
    mlp_dim = 384
    num_layers = 6
    num_classes = 10  # Number of classes for classification

    # Create a random input tensor with shape (6, 3, 256, 256)
    x = torch.randn(6, 3, 256, 256)
    print(f'The input shape: {x.shape}')

    # Initialize the Vision Transformer model
    model = VisionTransformer(in_channels, patch_size, embed_dim, num_heads, mlp_dim, num_layers, num_classes)

    # Forward pass with attention weights
    output, attn_weights = model(x, return_attention_weights=True)
    print(f'The output shape for classification: {output.shape}')

    # Plot attention maps
    for layer, attn_layer_weights in enumerate(attn_weights):
        print(f"Layer {layer}:")
        for head, attn_head_weights in enumerate(attn_layer_weights):
            print(f"  Head {head}:")
            attention = attn_head_weights.squeeze(0).detach().cpu().numpy()
            attention_patch = np.mean(attention, axis=0)  # Average attention across all patches
            image = x[0, 0].detach().cpu().numpy()
            plot_attention_map(attention_patch, image, patch_size, layer, head)

    return "Test Succeeded"


if __name__ == "__main__":
    test_vision_transformer()