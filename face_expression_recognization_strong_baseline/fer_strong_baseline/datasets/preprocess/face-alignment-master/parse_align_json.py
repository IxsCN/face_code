import json

class AlignConfig:
  def __init__(self, filepath):
    in_f = open(filepath, "r")
    assert(in_f is not None)
    self.ctx = json.load(in_f)
    in_f.close()
    self.load = True
    self.is_color, self.width, self.height = self.get_img_info()
    if self.is_color == 1:
      self.channel = 3
    else:
      self.channel = 1
    self.norm_ratio = self.get_norm_ratio()
    self.center_ind = self.get_center_ind()
    self.fill_value, self.fill_with_value = self.get_fill_info()

  def get_img_info(self):
    assert(self.load == True)
    return self.ctx['image_info']['is_color'], \
	   self.ctx['image_info']['width'], \
	   self.ctx['image_info']['height']

  def get_norm_ratio(self):
    assert(self.load == True)
    return self.ctx['norm_ratio']
  
  def get_fill_info(self):    
    assert(self.load == True)
    return self.ctx['fill_value'], self.ctx['fill_with_value']

  def get_center_ind(self):
    assert(self.load==True)
    return self.ctx['center_ind']

if __name__ == '__main__':
  align_param = AlignConfig('align.json')
  print(align_param.get_fill_info())
  print(align_param.get_img_info())
  print(align_param.get_norm_ratio())
  print(align_param.center_ind)
  

