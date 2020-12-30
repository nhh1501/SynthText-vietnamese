import torch
import h5py
import torch.nn.parallel

from models import modules, net, resnet, densenet, senet
import numpy as np
from torchvision import transforms, utils
from PIL import Image
import glob
device  = 'cpu'

import cv2
def np_to_variable(x, is_cuda=True, dtype=torch.FloatTensor):
  v = torch.from_numpy(x).type(dtype)
  if is_cuda:
      v = v.cuda()
  return v

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}



transform = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
                                ])

def define_model(is_resnet, is_densenet, is_senet):
    if is_resnet:
        original_model = resnet.resnet50(pretrained = True)
        Encoder = modules.E_resnet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])
    if is_densenet:
        original_model = densenet.densenet161(pretrained=True)
        Encoder = modules.E_densenet(original_model)
        model = net.model(Encoder, num_features=2208, block_channel = [192, 384, 1056, 2208])
    if is_senet:
        original_model = senet.senet154(pretrained='imagenet')
        Encoder = modules.E_senet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])

    return model


model = define_model(is_resnet=True, is_densenet=False, is_senet=False)
sp = torch.load('model_resnet',map_location=torch.device('cpu'))
for k, v in model.state_dict().items():
    try:
      if ('module.' + k) in sp:
        param = sp['module.' + k]
        v.copy_(param)
        # print(v.size())
      else:
        # print(k)
        # print(v.size())

        v.copy_(torch.randn(v.size()))
    except:
      import traceback
      traceback.print_exc()
# model.load_state_dict(sp)

model.to(device)
model.eval()
output_file= \
    'depth.h5'

list = glob.glob('img_bg/*jpg')
idx = 0
with torch.no_grad():
    for x in list:

        image = Image.open(x)
        # depth = Image.open(x)

        # sample = {'image': image, 'depth': depth}
        image = transform(image).to(device)
        # image = al['image']
        # depth = al['depth']
        # input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        # cv2.imshow("fdfdf",input)
        # cv2.waitKey(0)
        # input_var = np_to_variable(input,is_cuda=False)

        # input_var = input_var.permute(2,0,1).unsqueeze(0)

        image = torch.autograd.Variable(image, volatile=True)
        # depth = torch.autograd.Variable(depth, volatile=True)
        # depth = depth.unsqueeze(0)

        output = model(image.unsqueeze(0))

        # output = torch.nn.functional.upsample(output, size=[depth.size(2),depth.size(3)], mode='bilinear')
        output = torch.nn.functional.upsample(output, size=[512,512], mode='bilinear')

        pred_depth_image = output[0].data.squeeze().cpu().numpy().astype(np.float32)

        # pred_depth_image /= np.max(pred_depth_image)
        # pred_depth_image =pred_depth_image*255
        # imgplot = plot.imshow(pred_depth_image)
        # plot.show()
        # idx = idx + 1
        #
        # plot.imsave('Test_pred_depth_{:05d}.png'.format(idx), pred_depth_image, cmap="viridis")
        # print('idx', idx, 'saved')
        dbo = h5py.File(output_file, 'a')
        mask_dset = dbo.create_dataset(x.split('/')[1], data=pred_depth_image)
