import argparse
import yaml
import os
import json
import numpy as np
from PIL import Image
import pycocotools.mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import sys
sys.path.append('.')
from datasets import reader
import models
import inference as infer
import utils
from tqdm import tqdm
import cv2
import torchvision.transforms as transforms
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-num', default=-1, type=int)
    parser.add_argument('--load_model', default='/home/dell/下载/3090_deocclusion/deocclusion-master/experiments/Grap/new_experiment/pc_m/checkpoints/ckpt_iter_15000.pth.tar', type=str)
    parser.add_argument('--config', default='/home/dell/下载/3090_deocclusion/deocclusion-master/experiments/Grap/new_experiment/pc_m/config.yaml', type=str)
    parser.add_argument('--output', default=None, type=str)
    args = parser.parse_args()
    return args

def main(args):
    with open(args.config) as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

    for k, v in config.items():
        setattr(args, k, v)

    if not hasattr(args, 'exp_path'):
        args.exp_path = os.path.dirname(args.config)

    tester = Tester(args)
    tester.run()


class Tester(object):
    def __init__(self, args):
        self.args = args
        self.transform_x = transforms.Compose([
            transforms.Resize([320,320], Image.ANTIALIAS),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.5],std=[0.5])
        ])

    def prepare_data(self):
        dir_path = '/home/dell/下载/3090_deocclusion/deocclusion-master/data/Grap/putao/test_for_predict/'
        RD = GrapMask_read_dataset(dir_path)
        image_name_list = RD.get_name_pair()
        image_pairs =[]
        for i in range (len(image_name_list)-1):
            if i%2 == 0:
             image_pairs.append([dir_path+image_name_list[i],dir_path+image_name_list[i+1]])
        return image_pairs

    def prepare_model(self):
        self.model = models.__dict__[self.args.model['algo']](self.args.model, dist_model=False,
        boundary_out = args.model['boundary_for_output'],
        boundary_with_shared = args.model['boundary_shared']
        )
        self.model.load_state(self.args.load_model)
        self.model.switch_to('eval')

    def run(self):
        image_dir_pairs = self.prepare_data()
        self.prepare_model()
        #self.infer(image_dir_pairs)
        self.infer_formask()

    def GrayandBinary(self,image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image[image != 0] = 255.
        return image

    def read_image_and_process_forpredict(self,pair_path):
        image_path1 = pair_path[3][0]
        image_path2 = pair_path[3][1]
        print(image_path1)
        image = cv2.imread(image_path1)
        image2 = cv2.imread(image_path2)
        if image.shape != image2.shape:
            raise Exception("读取的两张图不能够组成图像对")
        else:
           image =  self.GrayandBinary(image)
           image2 = self.GrayandBinary(image2)

           image = Image.fromarray(image)
           image = self.transform_x(image)

           image2 = Image.fromarray(image2)
           image2 = self.transform_x(image2)
           return image, image2

    def infer(self,image_path):
        image1, image2 = self.read_image_and_process_forpredict(image_path)
        image1 = image1.cuda()
        image2 =image2.cuda()
        output = self.model.model(torch.cat([image1,image2], dim=1).view(-1,2,320,320))
        #print(output[:,:1,:,:])
        comp = output.argmax(dim=1, keepdim=True).float().view(1,320,320)
        comp[image2 == 0] = (image1 > 0).float()[image2 == 0]
        cv2.imwrite('./1.jpg', comp.view(320,320).cpu().numpy())

    def infer_formask(self):
        #image1, image2 = self.read_image_and_process_forpredict(image_path)

        image1= cv2.imread('/home/dell/下载/3090_deocclusion/deocclusion-master/data/Grap/putao/test_for_predict/4852/mask1.png',0)
        image2 = cv2.imread('/home/dell/下载/3090_deocclusion/deocclusion-master/data/Grap/putao/test_for_predict/4852/mask2.png',0)

        h,w = image1.shape


        image1[image1!=0]=255
        image2[image2 != 0] = 255

        image_out = image2+image1
        cv2.imwrite('./1.jpg',image_out)
        from PIL import Image


        image1 = self.transform_x(Image.fromarray(image1))
        image2 = self.transform_x(Image.fromarray(image2))

        image1= tensor_clamp(image1)
        image2 = tensor_clamp(image2)

        image1 = image1.cuda()
        image2 = image2.cuda()
        output = self.model.model(torch.cat([image2, image1], dim=1).view(-1, 2, 320, 320))
            # print(output[:,:1,:,:])
        comp = output.argmax(dim=1, keepdim=True).float().view(1, 320, 320)
        comp[image1 == 0] = (image2 > 0).float()[image1 == 0]
        #cv2.imwrite('./1.jpg', comp.view(320, 320).cpu().numpy())

        out = comp.cpu().numpy()
        max_value = out.max()
        out = out*255/max_value
        out_image = np.uint8(out)
        out_image = out_image.transpose(1,2,0)
        out_image= cv2.resize(out_image,(w,h))
        cv2.imwrite('./out_image.png',out_image)
        #cv2.imshow('1',out_image)
        #cv2.waitKey(0)

        # import matplotlib.pyplot as plt
        # plt.plot()
        # plt.imshow(comp.view(320,320).cpu())
        # plt.show()





import os
class GrapMask_read_dataset(object):
    def __init__(self,dataset_path):
        self.dataset_path = dataset_path

    def get_name_pair(self):
        image_name_list=[]
        for files in os.listdir(self.dataset_path):
            image_name_list.append(files)
        return image_name_list

def tensor_clamp(tensor_in):
        if torch.is_tensor(tensor_in):
            tensor_in[tensor_in > 0.1] = 1.
            tensor_in[tensor_in <= 0.1] = 0.
        else:
            if isinstance(tensor_in, list):
                for i in range(len(tensor_in)):
                    tensor_clamp(tensor_in[i])
            else:
                assert ("数据类型错误，应该是tensor 或者list")
        return tensor_in


if __name__ == "__main__":
    args = parse_args()
    main(args)
