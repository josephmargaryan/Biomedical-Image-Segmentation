import torch
import torch.nn as nn
from einops import rearrange
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import torchvision.transforms.v2 as v2

class Embedding(nn.Module):
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
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_embed = None

    def forward(self, x):
        batch_size, num_patches, embed_dim = x.size()
        if self.pos_embed is None or self.pos_embed.size(1) != num_patches:
            self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim, device=x.device))
        return x + self.pos_embed

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
        self.pos_encoding = PositionalEncoding(embed_dim)
        self.cls_token = ClassificationToken(embed_dim)
        self.encoder = nn.ModuleList([TransformerEncoderLayer(embed_dim, num_heads, mlp_dim) for _ in range(num_layers)])
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

class AttentionMaps:
    def __init__(self, model, file_path, img_size):
        self.model = model
        self.file_path = file_path
        self.img_size = img_size
        
    @staticmethod
    def plot_attention_map(attention, image, patch_size, layer, head=None):
        num_patches = int(np.sqrt(attention.shape[-1]))
        attention = attention.reshape(num_patches, num_patches)
        attention_resized = np.kron(attention, np.ones((patch_size, patch_size)))

        plt.imshow(image, cmap='gray')
        plt.imshow(attention_resized, cmap='jet', alpha=0.5)  # overlay attention map
        title = f"Layer {layer}"
        if head is not None:
            title += f", Head {head}"
        plt.title(title)
        plt.colorbar()
        plt.show()

    def VisualizeAttentionMaps(self):
        # Preprocess image
        img = Image.open(self.file_path).convert('RGB')
        img = v2.Compose([
            v2.Resize((self.img_size, self.img_size)),
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True),
            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])(img).view(1, 3, self.img_size, self.img_size)
        self.model.eval()
        out, attn_weights = self.model(img, return_attention_weights=True)
        # Average attention weights across all heads for each layer
        for layer in range(len(attn_weights)):
            attention = attn_weights[layer].mean(dim=1).squeeze(0).detach().cpu().numpy()

            # Take one patch's attention distribution
            attention_patch = attention[1:] # Remove CLS token 

            # Rescale attention to the original image size for visualization
            image = img[0, 0].detach().cpu().numpy()

            self.plot_attention_map(attention=attention_patch, image=image, patch_size=4, layer=layer)
        
    def VisualizeImg(self):
        img = Image.open(self.file_path)
        img = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True),
            v2.Resize((self.img_size, self.img_size))
            ])(img).permute(1, 2, 0)
        img = np.array(img)
        plt.imshow(img)
        plt.title('Original Image') 
        plt.axis('off')  
        plt.show()


def test():
    file_path = 'DRIVE/training/images/21_training.tif'    
    model = VisionTransformer(in_channels=3,
                            patch_size=16,
                            embed_dim=96,
                            num_heads=3,
                            mlp_dim=384,
                            num_layers=6,
                            num_classes=10)
    testatt = AttentionMaps(model=model, file_path=file_path, img_size=224)
    testatt.VisualizeAttentionMaps()
    testatt.VisualizeImg()

    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    print(f"Input shape: {x.shape}\nOutput shape: {out.shape}")
if __name__ == "__main__":
    test()