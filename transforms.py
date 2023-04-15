import albumentations as A
from config import Config

train_transform = A.Compose(
    [       A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # A.Resize(width=Config.width, height=Config.height)
                A.RandomBrightnessContrast(brightness_limit=1, contrast_limit=1, p=0.5)
            # A.Normalize(mean=Config.ADE_MEAN, std=Config.ADE_STD),   
    ]
)
val_transform = A.Compose(
    [      
            A.Resize(width=Config.width, height=Config.height),    
    #         A.Normalize(mean=Config.ADE_MEAN, std=Config.ADE_STD),   
    ]
)
