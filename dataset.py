from PIL import Image
import os
import PIL
import torch.utils.data as data
import numpy as np
import torchvision.transforms as transforms

class  AlignedDatset(data.Dataset):
    def __init__(self, image_dir, label_dir, image_size, train=True):
        self.train = train
        self.image_size = image_size
        if self.train:
            self.image_dir = image_dir
            self.images = os.listdir(image_dir)  
            
        self.label_dir = label_dir
        self.labels = os.listdir(label_dir)       
        
        #print(os.listdir(self.image_dir))
        
    def __len__(self):
        return len(self.label_dir)
    
    def get_transform(self, image_size, normalize=True):
        self.transform_list = []
        self.transform_list += [transforms.Resize((image_size, image_size*2))]
        self.transform_list += [transforms.ToTensor()]
        if normalize:
            self.transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        return transforms.Compose(self.transform_list)
    
    def __getitem__(self, index):
        
        
        if self.train:
            image_path = os.path.join(self.image_dir, self.images[index])
            image = Image.open(image_path).convert('RGB')  
            image_tensor = self.get_transform(self.image_size)(image).float()
        
        # label_path = os.path.join(self.label_dir, self.labels[index])
        # label = Image.open(label_path).convert('RGB')    
        # label_tensor = self.get_transform(self.image_size, normalize=False)(label).float()
        
        #return {'label': label_tensor.float(), 'image': image_tensor.float()}
        return {'image': image_tensor.float()}