import torch.nn as nn
from pathlib import Path
import numpy as np
import torch
from data.transforms import random_foreground_crop
import matplotlib.pyplot as plt
from utils.utils import load_nii
import os

class BinaryDataset(nn.Module):
    def __init__(self, data_root="dataset/binary_64x64", transform=None, overfit=False, split=True, phase="train", no_samples=200):
        super().__init__()
        self.data_root = data_root
        self.images = self.__load_files(data_root)
        print(f"Found {len(self.images)} images in {data_root}")
        self.skel = self.__load_files(data_root)
        print(f"Found {len(self.skel)} skeletons in {data_root}")
        self.phase = phase
        if overfit or split:
            if phase == "val":
                self.images = self.images[no_samples:]
                self.skel = self.skel[no_samples:]
            else:
                self.images = self.images[0:no_samples]
                self.skel = self.skel[0:no_samples]
        self.transform=transform

    @classmethod
    def __load_files(cls, path):
        
        """Load's the files or single file
        @path_or_file: Following types are supported, file name or path as string or single path name
        @load_only_one_path_idx: If multiple files or path is provided, this can be set to load single file
        """
        
        #path = Path(path)
        #print(f"Loading files from {path}")
        files_paths = sorted(os.listdir(path))#sorted(path.glob('*.nii.gz'))
        #print(f"Found {len(files_paths)} files in {path}", files_paths)
        #files_paths = [file for file in files_paths]
        return files_paths
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self,idx):
        image_path = self.images[idx]
        image = np.load(image_path)
        image = torch.from_numpy(image).float()
        image = image.unsqueeze(0)
        skel = None
        if self.skel:
            image_skel = self.skel[idx]
            skel = np.load(image_skel)
            skel = torch.from_numpy(skel).float()
            skel = skel.unsqueeze(0)
        if self.transform:
            image, skel = self.transform(image, skel)
        return image, skel, str(image_path.stem)

    def plot_sample(self, idx):
        img, skel, name = self.__getitem__(idx)

        if len(img.shape) == 4:  # Assuming shape is (C, D, H, W)
            fig = plt.figure(figsize=(12, 6))
        
            # Voxel plot for 3D data
            ax = fig.add_subplot(121, projection='3d')
            ax.voxels(img.squeeze(), facecolors='blue', edgecolor='k')
            ax.set_title('Original Image')
            ax.axis('off')

            ax2 = fig.add_subplot(122, projection='3d')
            ax2.voxels(skel.squeeze(), facecolors='red', edgecolor='k')
            ax2.set_title('Skeleton Image')
            ax2.axis('off')

        else:  # 2D plot
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(img.squeeze(), cmap='gray')
            axes[0].axis('off')
            axes[0].set_title('Original Image')
            axes[1].imshow(skel.squeeze(), cmap='gray')
            axes[1].axis('off')
            axes[1].set_title('Skeleton Image')

        fig.suptitle(name, fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()

class BezierDataset(BinaryDataset):
    def __init__(self, data_root="datasets/binary_bezier_256x256", transform=None, overfit=True, split=True, crop=True, phase="train", no_samples=4950):
        super(BezierDataset, self).__init__(data_root=data_root, transform=transform, overfit=overfit, phase= phase) 
        self.crop = crop
        self.images = self.__load_files(data_root, prefix="configuration")
        self.skel = self.__load_files(data_root, prefix="skel")
        if overfit or split:
            if phase == "val":
                self.images = self.images[no_samples:]
                self.skel = self.skel[no_samples:]
            else:
                self.images = self.images[0:no_samples]
                self.skel = self.skel[0:no_samples]

    @classmethod
    def __load_files(cls, path, prefix="configuration"):
        
        """Load's the files or single file
        @path_or_file: Following types are supported, file name or path as string or single path name
        @load_only_one_path_idx: If multiple files or path is provided, this can be set to load single file
        """
        
        path = Path(path)
        files_paths = sorted(path.glob('{}_*.npy'.format(prefix)))
        files_paths = [file for file in files_paths]
        return files_paths

    def __getitem__(self,idx):
        image_path = self.images[idx]
        image = np.load(image_path).squeeze()
        image = torch.from_numpy(image).float()
        image = image.unsqueeze(0)
        skel = None
        if self.skel:
            image_skel = self.skel[idx]
            skel = np.load(image_skel).squeeze()
            skel = torch.from_numpy(skel).float()
            skel = skel.unsqueeze(0)
        if self.crop:
            image, skel = random_foreground_crop(image, skel)
        if self.transform:
            image, skel = self.transform(image, skel)
        return image, skel, str(image_path.stem)

class BezierDataset3D(BezierDataset):
    def __init__(self, data_root="/projects/rectal_response/small_bowel/nnUNet_raw/Dataset801_Fold1/labelsTr", transform=None, overfit=True, split=True, crop=True, phase="train", no_samples=20):
        super(BezierDataset3D, self).__init__(
            data_root=data_root, 
            transform=transform, 
            overfit=overfit, 
            phase=phase, 
            crop=crop, 
            split=split, 
            no_samples=no_samples
        ) 
        self.images = self.__load_files(data_root)
        self.skel = self.__load_files(data_root)
    @classmethod
    def __load_files(cls, path):
        
        """Load's the files or single file
        @path_or_file: Following types are supported, file name or path as string or single path name
        @load_only_one_path_idx: If multiple files or path is provided, this can be set to load single file
        """
        
        #path = Path(path)
        #print(f"Loading files from {path}")
        files_paths = sorted(os.listdir(path))#sorted(path.glob('*.nii.gz'))
        #print(f"Found {len(files_paths)} files in {path}", files_paths)
        #files_paths = [file for file in files_paths]
        return files_paths
    def __getitem__(self,idx):
        image_path = os.path.join (self.data_root, self.images[idx])
        image,affine = load_nii(image_path)#np.load(image_path).squeeze()
        #image = torch.from_numpy(image).float()
        #image = image.unsqueeze(0)
        skel = None
        if self.skel:
            image_skel = self.skel[idx]
            image_skel = os.path.join("/home/l.cai/small_bowel/small_bowel_data_code/ex_preprocess/all_mri_2labels_connect18",image_skel.split('/')[-1])
            skel, affine = load_nii(image_skel)#np.load(image_skel).squeeze()
            skel[skel==1] = 0
            skel[skel==2] = 1
            # print(skel.shape)
            #skel = torch.from_numpy(skel).float()
            #skel = skel.unsqueeze(0)
        if self.crop:
            image, skel = random_foreground_crop(image.squeeze(0), skel.squeeze(0), out_size=64)
        if self.transform:
            transformed = self.transform({"image": image, "skel": skel})
            image = transformed["image"]
            skel = transformed["skel"]
        #print(image.shape, skel.shape)
        return {'image':image, "skel":skel, "image_path":image_path}#image, skel, str(image_path.stem)