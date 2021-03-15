import torch
from torch.utils.data import Dataset
# torch.multiprocessing.set_start_method('spawn')
from PIL import Image
from torchvision import transforms
from numpy import shape

import os
#path = "F:/Cuteness AI/Code/Trainable_Cuteness-main/"
#os.chdir(path) # changing to the file directory we want

class ImageDataSet(Dataset):
    '''Dataset for loading images and associated rank labels'''
    def __init__(self, 
                 image_file_list,
                 image_size, 
                 device=torch.device('cpu'),
                 do_augmentation=True):
        '''Instantiation of input data and augmentation transforms'''

        self.device = device
        self.image_files = image_file_list['ImageFile']
        self.binary_rank = image_file_list['Rating']
        self.image_size = image_size
        self.do_augmentation = do_augmentation
        self.resize_transform = transforms.Resize((self.image_size,
                                                   self.image_size))

        augmentations = [transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomAffine(degrees = 45,
                                                translate=(0.2,0.2),
                                                scale=(0.7,1.3)),
                        transforms.ColorJitter(brightness=0.05,
                                                contrast=0.05,
                                                saturation=0.05,
                                                hue=0.05)]

        self.augmentation_transforms = transforms.Compose(augmentations)
        normal_params = (0.485,0.456,0.406),(0.229,0.224,0.225) #VGG mean/std
        required = [transforms.ToTensor(),
                    transforms.Normalize(normal_params[0],normal_params[1])] 
        self.required_transforms = transforms.Compose(required)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        '''Return an image and rank value from the dataset

        Loads the image using PIL.Image and then makes sure that it is an RGB
        image. If it is RGBA it will convert it back to RGB. It then applies any
        transforms and augmentations required, converts the image and rank to
        tensor, and returns them both.
        '''

        def pure_pil_alpha_to_color_v2(image, color=(255, 255, 255)):
            """Alpha composite an RGBA Image with a specified color.

            Source: http://stackoverflow.com/a/9459208/284318

            Keyword Arguments:
            image -- PIL RGBA Image object
            color -- Tuple r, g, b (default 255, 255, 255)

            """
            image.load()  # needed for split()
            background = Image.new('RGB', image.size, color)
            background.paste(image, mask=image.split()[3])
            return background

        image = Image.open('Data/raw-img/'+self.image_files[idx])
        image_shape = shape(image)
        if len(image_shape) == 3:
            if image_shape[2] == 4:
                image = pure_pil_alpha_to_color_v2(image)
        else:
            image = image.convert('RGB')
        image = self.resize_transform(image)
        if self.do_augmentation:
            image = self.augmentation_transforms(image)
        image = self.required_transforms(image)
        rank = torch.Tensor([self.binary_rank[idx]]).float()
        return image, rank
