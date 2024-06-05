import torch
import numpy as np
import torchvision
import torchvision.transforms.v2 as v2
from PIL import Image
import matplotlib.pyplot as plt
from dataset import device
from AttentionUnet import AttentionUNet
from inference_utils import inference

"""
Functionality to extratc feature maps learned by the AttentionUnet and stored in the channel dimenions
"""

def get_intermediate_outputs(model, x):
    outputs = {}
    
    # Forward pass
    e1 = model.Conv1(x)
    outputs['Conv1'] = e1

    e2 = model.MaxPool(e1)
    e2 = model.Conv2(e2)
    outputs['Conv2'] = e2

    e3 = model.MaxPool(e2)
    e3 = model.Conv3(e3)
    outputs['Conv3'] = e3

    e4 = model.MaxPool(e3)
    e4 = model.Conv4(e4)
    outputs['Conv4'] = e4

    e5 = model.MaxPool(e4)
    e5 = model.Conv5(e5)
    outputs['Conv5'] = e5

    d5 = model.Up5(e5)
    s4 = model.Att5(gate=d5, skip_connection=e4)
    d5 = torch.cat((s4, d5), dim=1)
    d5 = model.UpConv5(d5)
    outputs['UpConv5'] = d5

    d4 = model.Up4(d5)
    s3 = model.Att4(gate=d4, skip_connection=e3)
    d4 = torch.cat((s3, d4), dim=1)
    d4 = model.UpConv4(d4)
    outputs['UpConv4'] = d4

    d3 = model.Up3(d4)
    s2 = model.Att3(gate=d3, skip_connection=e2)
    d3 = torch.cat((s2, d3), dim=1)
    d3 = model.UpConv3(d3)
    outputs['UpConv3'] = d3

    d2 = model.Up2(d3)
    s1 = model.Att2(gate=d2, skip_connection=e1)
    d2 = torch.cat((s1, d2), dim=1)
    d2 = model.UpConv2(d2)
    outputs['UpConv2'] = d2

    out = model.Conv(d2)
    outputs['Output'] = out

    return outputs


def visualize_feature_maps(feature_maps, layer_name, num_maps=6):
    feature_maps = feature_maps.cpu().detach().numpy()
    num_channels = feature_maps.shape[1]
    
    fig, axes = plt.subplots(1, num_maps, figsize=(15, 15))
    for i in range(num_maps):
        if i < num_channels:
            ax = axes[i]
            ax.imshow(feature_maps[0, i, :, :], cmap='viridis')
            ax.axis('off')
            ax.set_title(f'{layer_name} Map {i+1}')
    plt.show()



def test():
    # Preprocess the input image
    img = Image.open("/kaggle/input/thyroid-nodules/masks_folder/T02_case_0_slice_4.png").convert('RGB')
    img = torchvision.transforms.functional.crop(img=img, top=40, left=0, height=273, width=210)
    input_image = v2.Compose([
        v2.Resize((192, 192)),
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])(img).view(1, 3, 192, 192).to(device)

    # Load the model

    model = AttentionUNet()
    # model.load_state_dict(torch.load('model.pth')) # Load pretrained weights
    model.eval() # Set model to eval mode

    # Get intermediate outputs
    intermediate_outputs = get_intermediate_outputs(model, input_image)

    # Visualize feature maps from different layers
    for layer_name, feature_map in intermediate_outputs.items():
        visualize_feature_maps(feature_map, layer_name, num_maps=6)


    # Optional: Visualize the original image and model prediction
    inference(model, "/kaggle/input/thyroid-nodules/masks_folder/T02_case_0_slice_4.png")