from unet import UNET
from dataset import val_loader
import matplotlib.pyplot as plt
import torch 
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from dataset import device, val_loader
import torchvision
import torchvision.transforms.v2 as v2
from PIL import Image


def plot_predictions(model, val_loader, device):
    model.eval()
    fig, axs = plt.subplots(4, 3, figsize=(15, 30))

    for i, (x, y) in enumerate(tqdm(val_loader)):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            outputs = torch.sigmoid(model(x)).cpu().numpy()
            outputs = (outputs > 0.5).astype(float)
            images = x.cpu().numpy()

            for j in range(min(6, x.size(0))):  # Iterate over each sample in the batch
                output_single = outputs[j, 0]  # Assuming output has shape (batch_size, 1, height, width)
                image_single = images[j].transpose(1, 2, 0)  # Move channel dimension to last

                # Rescale values to [0, 1] range
                image_single = (image_single + 1) / 2  # Assuming input was normalized to [-1, 1]

                axs[j, 0].imshow(image_single)  # Display RGB image
                axs[j, 0].set_title('Original Image')
                axs[j, 0].axis('off')

                axs[j, 1].imshow(y[j, 0].cpu().numpy(), cmap='gray')  # Display ground truth mask
                axs[j, 1].set_title('Ground Truth Mask')
                axs[j, 1].axis('off')

                axs[j, 2].imshow(output_single, cmap='gray')  # Display predicted mask
                axs[j, 2].set_title('Predicted Mask')
                axs[j, 2].axis('off')

    plt.show()

def inference(model, file_path):
    img = Image.open(file_path).convert('RGB')  # Load image
    # Apply preprocessing steps
    img = torchvision.transforms.functional.crop(img=img, top=40, left=0, height=img.height-50, width=img.width)
    img = torchvision.transforms.functional.resize(img=img, size=(192, 192))
    img_normalized = torchvision.transforms.functional.to_tensor(img)
    img_normalized = torchvision.transforms.functional.normalize(img_normalized, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    img = img_normalized.unsqueeze(0).to(device)  # Add batch dimension and move to device

    model.eval()
    with torch.no_grad():
        out = torch.sigmoid(model(img))  # Forward pass through the model and apply sigmoid activation
        prediction = (out > 0.5).float()  # Threshold predictions at 0.5

    # Convert tensors to numpy arrays for visualization
    original_image = img_normalized.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    original_image_unnormalized = (original_image + 1) / 2  # Unnormalize for clearer visualization
    prediction_image = prediction.squeeze(0).cpu().numpy().transpose(1, 2, 0)

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))  # Create subplots
    ax[0].imshow(original_image_unnormalized)  # Display unnormalized original image
    ax[0].set_title('Original Image')
    ax[0].axis('off')  # Turn off axis labels
    ax[1].imshow(prediction_image, cmap='gray')  # Display prediction
    ax[1].set_title('Prediction')
    ax[1].axis('off')  # Turn off axis labels
    plt.show()  # Show the plot


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
    
