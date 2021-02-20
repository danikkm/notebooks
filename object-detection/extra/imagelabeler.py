import os
from glob import glob
import pandas as pd

class ImageLabeler:
  def __init__(self, class_names=[''], file_types=[''], train_path = '', test_path = ''):        
    if len(class_names) == 0:
      raise ValueError("Array class_names with length >=1 was expected")
    else:
      self.class_names = class_names
    if len(file_types) == 0:
      raise ValueError("Array file_types with length >=1 was expected")
    else:
      self.file_types = file_types
    if not train_path or not test_path:
      raise ValueError("Path for the train and test directories requried")
    else:
      self.train_path = train_path
      self.test_path = test_path
  
  def create_dfs(self):
    if train_path == os.path.abspath(os.getcwd()) or test_path == os.path.abspath(os.getcwd()):
      assert False, 'Please provide path to the folder'
    
    train_dfs = [self.__create_df(path=self.train_path, class_name=class_name, file_types=self.file_types)
           for class_name in class_names]
    test_dfs = [self.__create_df(path=self.test_path, class_name=class_name, file_types=self.file_types)
          for class_name in class_names]
    
    return pd.concat(train_dfs), pd.concat(test_dfs)
  
  def write_to_csv(self, path ='', train_dfs=[], test_dfs=[]):
    if not path:
      path = os.path.abspath(os.path.join(self.train_path, os.pardir))
      print(path)
    if train_dfs.empty or test_dfs.empty:
      assert False, 'Please provide train and test dataframes'
      
    train_dfs.to_csv(path+'/train_labels.csv', index=False)
    test_dfs.to_csv(path+'/test_labels.csv', index=False)
  
  def __create_df(self, path='', class_name='', file_types = ['']):
    result = []
    for f_path in glob(path + class_name +'/**', recursive=True):
      f_name = os.path.basename(f_path)
      f_ext = f_name.split(".")[-1].lower()

      if f_ext in file_types:
        result.append([f_name, class_name])

    return pd.DataFrame(result, columns=['f_name', 'label'])