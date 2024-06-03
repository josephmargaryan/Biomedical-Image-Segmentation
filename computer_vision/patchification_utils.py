import torch
import torchvision
import matplotlib.pyplot as plt
from dataset import val_set

class PatchificationUtils:
    @staticmethod
    def img_to_patch(x, patch_size, flatten_channels=True):
        """
        Args:
            x: Tensor representing the image of shape [B, C, H, W]
            patch_size: Number of pixels per dimension of the patches (integer)
            flatten_channels: If True, the patches will be returned in a flattened format
                               as a feature vector instead of a image grid.
        """
        B, C, H, W = x.shape
        x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
        x = x.flatten(1, 2)  # [B, H'*W', C, p_H, p_W]
        if flatten_channels:
            x = x.flatten(2, 4)  # [B, H'*W', C*p_H*p_W]
        return x

    @staticmethod
    def plot_image_examples(val_set, num_images=4):
        """
        Plot image examples from the validation set.

        Args:
            val_set: Validation dataset.
            num_images: Number of images to plot. Default is 4.
        """
        images = torch.stack([val_set[idx][0] for idx in range(num_images)], dim=0)

        img_grid = torchvision.utils.make_grid(images, nrow=4, normalize=True, pad_value=0.9)
        img_grid = img_grid.permute(1, 2, 0)

        plt.figure(figsize=(8, 8))
        plt.title("Image examples of dataset")
        plt.imshow(img_grid)
        plt.axis("off")
        plt.show()
        plt.close()

    @staticmethod
    def visualize_image_patches(val_set, patch_size, image_size, num_images=4):
        """
        Visualize images as input sequences of patches.

        Args:
            images: Tensor representing the images of shape [B, C, H, W].
            patch_size: Number of pixels per dimension of the patches (integer).
            image_size: Size of the input image (integer).
        """
        images = torch.stack([val_set[idx][0] for idx in range(num_images)], dim=0)
        img_patches = PatchificationUtils.img_to_patch(images, patch_size=patch_size, flatten_channels=False)

        fig, ax = plt.subplots(images.shape[0], 1, figsize=(14, 12))
        fig.suptitle("Images as input sequences of patches")
        for i in range(images.shape[0]):
            img_grid = torchvision.utils.make_grid(img_patches[i], nrow=int(image_size/patch_size), normalize=True, pad_value=0.9)
            img_grid = img_grid.permute(1, 2, 0)
            ax[i].imshow(img_grid)
            ax[i].axis("off")
        plt.show()
        plt.close()

    @staticmethod
    def visualize_patches_from_dataset(val_set, patch_size=4, num_images=4):
        """
        Visualize images from a dataset as input sequences of patches.

        Args:
            val_set: Dataset containing images.
            patch_size: Number of pixels per dimension of the patches (integer).
            num_images: Number of images to visualize from the dataset (integer).
        """
        images = torch.stack([val_set[idx][0] for idx in range(num_images)], dim=0)
        img_patches = PatchificationUtils.img_to_patch(images, patch_size=patch_size, flatten_channels=True)

        fig, ax = plt.subplots(images.shape[0], 1, figsize=(14, 3))
        fig.suptitle("Images as input sequences of patches flattened")
        for i in range(images.shape[0]):
            img_grid = torchvision.utils.make_grid(img_patches[i], nrow=int(images.shape[2]/patch_size), normalize=True, pad_value=0.9)
            img_grid = img_grid.permute(1, 2, 0)
            ax[i].imshow(img_grid)
            ax[i].axis("off")
        plt.show()
        plt.close()

if __name__ == "__main__":
    # Call the plot_image_examples function directly
    PatchificationUtils.plot_image_examples(val_set, num_images=4)

    # Call the visualize_image_patches function directly
    PatchificationUtils.visualize_image_patches(val_set, patch_size=32, image_size=224, num_images=4)

    # Call the visualize_patches_from_dataset function directly
    PatchificationUtils.visualize_patches_from_dataset(val_set, patch_size=32, num_images=4)