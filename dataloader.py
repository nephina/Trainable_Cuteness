import torch
from torch.utils.data import Dataset
torch.multiprocessing.set_start_method('spawn')
from PIL import Image
from torchvision import transforms
from numpy import shape

class ImageDataSet(Dataset):
    # dataset for loading MRI input/mask pairs
    def __init__(self, 
                 image_file_list,
                 image_size, 
                 device=torch.device('cpu'),
                 do_augmentation=True):

        self.device = device
        self.image_files = image_file_list['ImageFile']
        self.binary_rank = image_file_list['Rating']
        self.image_size = image_size
        self.do_augmentation = do_augmentation
        self.resize_transform = transforms.Resize((self.image_size,
                                                   self.image_size))

        augmentations = [transforms.RandomHorizontalFlip(p=0.5),

                         transforms.RandomAffine(30,
                                                 translate=(0.3,0.3),
                                                 scale=(0.8,1.2)),

                         transforms.ColorJitter(brightness=0.05,
                                                contrast=0.05,
                                                saturation=0.05,
                                                hue=0.05)]

        self.augmentation_transforms = transforms.Compose(augmentations)

        required = [transforms.ToTensor(), transforms.Normalize(0,1)]
        self.required_transforms = transforms.Compose(required)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

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
        image = self.required_transforms(image).to(self.device)
        rank = torch.Tensor([self.binary_rank[idx]]).float().to(self.device)
        return image, self.binary_rank[idx]
