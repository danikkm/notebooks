import torchvision.models as models

class ModelLoader:
  def __init__(self, batch_size=64, epochs=32):
    self.model = None
    self.batch_size = batch_size
    self.epochs = epochs
  
  def load_mobilenetv2(self, pretrained=False, num_classes=10):
    self.model = models.mobilenet_v2(pretrained=False, num_classes=num_classes)
    self.name = 'mobilenetv2'
    self.checkpoint_path = f'./checkpoint/ckpt_{self.name}_bs{self.batch_size}_ep{self.epochs}.pth'
    
    return self.model, self.name, self.checkpoint_path
  
  def load_mobilenetv3(self, mode='large', pretrained=False, num_classes=10):
    if mode == 'large':
      self.model = models.mobilenet_v3_large(pretrained=False, num_classes=num_classes)
    else:
      self.model = models.mobilenet_v3_small(pretrained=False, num_classes=num_classes)

    self.name = f'mobilenetv3{mode}'
    self.checkpoint_path = f'./checkpoint/ckpt_{self.name}_bs{self.batch_size}_ep{self.epochs}.pth'
    
    return self.model, self.name, self.checkpoint_path  
