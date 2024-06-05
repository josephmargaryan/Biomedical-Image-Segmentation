import pandas as pd 
import numpy as np 
import torchvision
import torchvision.transforms.v2 as v2 
import torch
import glob
from PIL import Image
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

image_paths = sorted(glob.glob('/Users/josephmargaryan/Desktop/computer_vision/DRIVE/training/images/*.tif'))
mask_paths = sorted(glob.glob('/Users/josephmargaryan/Desktop/computer_vision/DRIVE/training/1st_manual/*.gif'))
df = pd.DataFrame(list(zip(image_paths, mask_paths)), columns=['image path', 'mask path'])

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image_path = self.df.iloc[idx]['image path']
        mask_path = self.df.iloc[idx]['mask path']
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        image = torchvision.transforms.functional.crop(img=image, top=10, left=0, height=564, width=565)
        mask = torchvision.transforms.functional.crop(img=mask, top=10, left=0, height=564, width=565)
        
        if self.transform:
            image, mask = self.transform(image, mask)
            image = v2.Compose([v2.ToImage(), v2.ToDtype(dtype=torch.float32, scale=True)])(image)
            image = v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image)
            mask = v2.Compose([v2.ToImage(), v2.ToDtype(dtype=torch.float32, scale=True)])(mask)
        return image, mask
    
    
transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.RandomHorizontalFlip(),
    v2.RandomPerspective(),
    v2.RandomRotation((-45, 45)),
    v2.RandomVerticalFlip(),
    v2.ElasticTransform()
])
dataset = CustomDataset(df=df, transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=6, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=6, shuffle=True)

class VisualizationUtils:
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def plot_grids(self, grids, titles=["Input", "Target"]):
        nrow = len(grids)
        fig = plt.figure(figsize=(8, nrow), dpi=300)
        fig.subplots_adjust(wspace=0, hspace=0)
        for i in range(1, nrow + 1):
            sub = fig.add_subplot(nrow, 1, i)
            sub.xaxis.set_visible(False)
            sub.set_yticks([])
            sub.set_ylabel(titles[i - 1], rotation=0, fontsize=15, labelpad=30)
            sub.imshow(grids[i - 1])
        plt.show()

    @staticmethod
    def create_grids(img_list, nrof_items, pad, norm=True):
        return list(map(lambda imgs: torchvision.utils.make_grid(imgs[:nrof_items], normalize=norm, padding=pad, nrow=nrof_items).permute(1, 2, 0), img_list))

    def create_segment_grids(self, nrof_items=5, pad=4):
        loader_iter = iter(self.dataloader)
        inp, target = next(loader_iter)
        return self.create_grids([inp, target], nrof_items, pad)


def test():
    for (x, y) in train_loader:
        print(x.shape)
        print(y.shape)
        break
    for (x, y) in train_loader:
        print(x.shape)
        print(y.shape)
        break
    visualizer = VisualizationUtils(val_loader)
    grids = visualizer.create_segment_grids(nrof_items=5, pad=4)
    visualizer.plot_grids(grids)
if __name__ == "__main__":
    test()