import PIL
import pandas as pd
import torch
import os
import cv2
from config import Config
from transformers import ViTImageProcessor, ViTForImageClassification
from torchvision.models import resnet50, ResNet50_Weights, efficientnet_v2_s, efficientnet_v2_S_Weights

class KedsDataset(torch.utils.data.Dataset):
    """Image segmentation dataset."""

    def __init__(self, transform, ids:pd.DataFrame, images_path:str='/content/images/'):
        """
        Args:
            dataset
        """
        self.data = ids 
        self.transform = transform
        self.images_path = images_path
        self.processor = efficientnet_v2_S_Weights.IMAGENET1K_V1.transforms()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
      
        row = self.data.iloc[idx]
        path_to_img = row['image']

        label = row['class_id']
        original_image= cv2.imread(os.path.join(self.images_path, path_to_img))
        
        
        # try:
        transformed = self.transform(image=original_image)
        # except:
        #   print(path_to_img, original_image)
      
        image = transformed['image'] 

        # convert to C, H, W
        # image = image.transpose(2,0,1)
        transformed = self.processor(PIL.Image.fromarray(image))
        # print(transformed.size())
        return transformed, label
