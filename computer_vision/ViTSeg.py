import torch
import torch.nn as nn
from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.v2 as v2
import torchvision
from PIL import Image
from dataset import device

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
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_embed = None

    def forward(self, x):
        batch_size, num_patches, embed_dim = x.size()
        if self.pos_embed is None or self.pos_embed.size(1) != num_patches:
            self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim, device=x.device))
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
        self.pos_encoding = PositionalEncoding(embed_dim)
        self.encoder = nn.ModuleList([TransformerEncoderLayer(embed_dim, num_heads, mlp_dim) for _ in range(num_layers)])

        # Calculate the number of upsampling layers required
        num_upsample_layers = int(torch.log2(torch.tensor(image_size // patch_size)))
        self.upsample_layers = nn.ModuleList()
        in_channels = embed_dim
        for _ in range(num_upsample_layers):
            out_channels = in_channels // 2 if in_channels > embed_dim // 4 else embed_dim // 4
            self.upsample_layers.append(UpsampleBlock(in_channels, out_channels))
            in_channels = out_channels

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

        for upsample_layer in self.upsample_layers:
            x = upsample_layer(x)

        x = self.segmentation_head(x)
        
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
        height, width = img.size
        img = torchvision.transforms.functional.crop(img=img, top=40, left=0, height=height-50, width=width)
        img = v2.Resize((self.img_size, self.img_size))(img)
        img = v2.Compose([v2.ToImage(), v2.ToDtype(dtype=torch.float32, scale=True)])(img)
        img = v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
        img = img.view(-1, 3, self.img_size, self.img_size).to(device)
        self.model.eval()
        out, attn_weights = self.model(img, return_attention_weights=True)
        # Average attention weights across all heads for each layer
        for layer in range(len(attn_weights)):
            attention = attn_weights[layer].mean(dim=1).squeeze(0).detach().cpu().numpy()

            # Take one patch's attention distribution
            attention_patch = attention  

            # Rescale attention to the original image size for visualization
            image = img[0, 0].detach().cpu().numpy()

            self.plot_attention_map(attention=attention_patch, image=image, patch_size=4, layer=layer)
        
    def VisualizeImg(self):
        img = np.array(Image.open(self.file_path))
        plt.imshow(img)
        plt.title('Original Image') 
        plt.axis('off')  
        plt.show()

def test():
    configs = {
        'in_channels': 3,
        'patch_size': 16,
        'embed_dim': 96,
        'num_heads': 4,
        'mlp_dim': 192,
        'num_layers': 6,
        'num_classes': 1,
        'image_size': 256
    }
    x = torch.randn(1, 3, 256, 256)
    model = VisionTransformerSegmentation(**configs)
    out = model(x)
    assert out.shape == (1, 1, 256, 256), f'Outputs expected size was: [1, 256, 256], but got: {out.shape}'
    print("Test was a success")

    file_path = "DRIVE/training/images/21_training.tif"
    att = AttentionMaps(model=model, file_path=file_path, img_size=224)
    att.VisualizeAttentionMaps()
    att.VisualizeImg()

if __name__ == "__main__":
    test()