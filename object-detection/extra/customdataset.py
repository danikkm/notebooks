import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class CustomDataset(Dataset):
  #TODO: add support for train and test csv's
  def __init__(self, csv_file, img_folder_dir, shape, mapping_dict={}, transform=None):
    self.anotations = pd.read_csv(os.path.abspath(os.getcwd()) + csv_file)
    self.img_folder_dir = os.path.abspath(os.getcwd()) + img_folder_dir
    self.shape = shape
    self.mapping_dict = mapping_dict
    self.transform = transform
  
  def __len__(self):
    return len(self.anotations)
  
  def __getitem__(self, index):
    img_path = os.path.join(self.img_folder_dir, self.anotations.iloc[index, 0])
    image = Image.open(img_path).convert('RGB')
    image = self.__resize(shape=self.shape, image=image)
    y_label = torch.tensor(int(self.__map_str_to_int(self.anotations.iloc[index, 1])))
    
    if self.transform:
      image = self.transform(image)
      
    return image, y_label
  
  def __resize(self, shape, image = None):
    new_image = image.resize(shape)
    return np.float32(new_image)
  
  # TODO: think of smth else
  def __map_str_to_int(self, label):
    return self.mapping_dict[label]
