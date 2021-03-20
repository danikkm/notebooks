from models.mobilenetv2 import MobileNetV2
from models.mobilenetv3 import MobileNetV3

class ModelLoader:
  def __init__(self, batch_size=64, epochs=32):
    self.model = None
    self.batch_size = batch_size
    self.epochs = epochs
  
  def load_mobilenetv2(self):
    self.model = MobileNetV2()
    self.name = 'mobilenetv2'
    self.checkpoint_path = f'./checkpoint/ckpt_{self.name}_bs{self.batch_size}_ep{self.epochs}.pth'
    
    return self.model, self.name, self.checkpoint_path
  
  def load_mobilenetv3(self, mode='large', classes_num=10, input_size=32, dropout=0.8, width_multiplier=1.0):
    self.model = MobileNetV3(mode=mode, classes_num=classes_num,
                 input_size=input_size, dropout=dropout,
                 width_multiplier=width_multiplier)
    self.name = 'mobilenetv3'
    self.checkpoint_path = f'./checkpoint/ckpt_{self.name}_bs{self.batch_size}_ep{self.epochs}.pth'
    
    return self.model, self.name, self.checkpoint_path  