import numpy as np
try:
    import mc
except Exception:
    pass
import cv2
import os
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import utils
from . import reader
import random
import cmath
import torch.nn.functional as F


def revise_mask(crop_mask):
 fg_image = np.zeros(crop_mask.shape, dtype=np.uint8)
 fg_image2 = np.ones(crop_mask.shape, dtype=np.uint8) * 255
 FG_img = np.where(crop_mask, fg_image, fg_image2)     #参数1为Ture返回参数2，否则参数3
 return FG_img


def CropRemove_WithOriginal(image,mask1,patten,occulde_ratio):
    #读取被遮挡图和掩码图像
    #image = cv2.cvColor(image,cv2.COLOR_BGR2GRAY)
    h1,w1,c = image.shape
    #将掩码图像转化为与被遮挡图一样的大小（确保尺度差异较大对遮挡效果的影响）
    mask1 = cv2.resize(mask1,(w1,h1))
    h2,w2,c = mask1.shape
    if patten ==1:
        cropw = int(w1 * occulde_ratio)
        new_w = w1+w2-cropw
        new_h = h1
        intact_image = np.zeros((new_h,new_w,3),np.uint8)
        mask_image = intact_image.copy()
        intact_image[:h1,:w1,:] = image
        mask_image[:,w1-cropw:,:] = mask1

        blank_image = np.zeros((new_h, new_w, 3), np.uint8)
        mask_crop = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)
        mask_crop[mask_crop != 0] = 255
        mask_crop = revise_mask(mask_crop)
        mask_crop = mask_crop[:,:cropw]
        mask_for = np.zeros((h1, w1), np.uint8)
        mask_for.fill(255)
        mask_for[:,w1-cropw:]= mask_crop
        mask = np.zeros_like(image)
        # mask = mask.transpose(0, 2, 3, 1).squeeze(0)
        for i in range(mask.shape[2]):
            mask[:, :, i] = mask_for
        c = cv2.bitwise_and(image, mask)
        blank_image[:h1,:w1,:] = c
        return blank_image,mask_image,intact_image
    elif patten == 2:
        ratio = abs(cmath.sqrt(occulde_ratio))
        croph = int(h1*ratio)
        cropw = int(w1*ratio)

        new_h = h1+h2 - croph
        new_w = w1+w2 - cropw

        intact_image = np.zeros((new_h, new_w, 3), np.uint8)
        mask_image = intact_image.copy()
        mask_image[:h2,w1-cropw:,:] = mask1
        intact_image[h2-croph:, :w1, :] = image



        blank_image = np.zeros((new_h, new_w, 3), np.uint8)
        mask_crop = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)
        mask_crop[mask_crop != 0] = 255
        mask_crop = revise_mask(mask_crop)
        mask_crop = mask_crop[h2-croph:,:cropw]

        mask_for = np.zeros((h1, w1), np.uint8)
        mask_for.fill(255)
        mask_for[:croph,w1-cropw:]= mask_crop

        mask = np.zeros_like(image)
        for i in range(mask.shape[2]):
            mask[:, :, i] = mask_for
        c = cv2.bitwise_and(image, mask)
        blank_image[h2-croph:, :w1, :] = c
        return blank_image,mask_image, intact_image
    elif patten == 3:
        croph = int(h1* occulde_ratio)
        cropw = w1

        new_h = h1+h2 - croph
        new_w = w1

        intact_image = np.zeros((new_h, new_w, 3), np.uint8)
        mask_image = intact_image.copy()
        mask_image[:h2, :, :] = mask1
        intact_image[h2-croph:, :w1, :] = image

        blank_image = np.zeros((new_h, new_w, 3), np.uint8)
        mask_crop = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)
        mask_crop[mask_crop != 0] = 255
        mask_crop = revise_mask(mask_crop)
        mask_crop = mask_crop[h2 - croph:, :]

        mask_for = np.zeros((h1, w1), np.uint8)
        mask_for.fill(255)
        mask_for[:croph, :] = mask_crop

        mask = np.zeros_like(image)
        for i in range(mask.shape[2]):
            mask[:, :, i] = mask_for
        c = cv2.bitwise_and(image, mask)
        blank_image[h2 - croph:, :, :] = c
        return blank_image,mask_image, intact_image
    elif patten == 4:
        ratio = abs(cmath.sqrt(occulde_ratio))
        croph = int(h1 * ratio)
        cropw = int(w1 * ratio)
        new_h = h1+h2 - croph
        new_w = w1+w2 - cropw

        intact_image = np.zeros((new_h, new_w, 3), np.uint8)
        mask_image = intact_image.copy()
        mask_image[:h2, :w2, :] = mask1
        intact_image[h2-croph:, w2-cropw:, :] = image

        blank_image = np.zeros((new_h, new_w, 3), np.uint8)
        mask_crop = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)
        mask_crop[mask_crop != 0] = 255
        mask_crop = revise_mask(mask_crop)
        mask_crop = mask_crop[h2 - croph:,w2-cropw :]

        mask_for = np.zeros((h1, w1), np.uint8)
        mask_for.fill(255)
        mask_for[:croph, :cropw] = mask_crop

        mask = np.zeros_like(image)
        for i in range(mask.shape[2]):
            mask[:, :, i] = mask_for
        c = cv2.bitwise_and(image, mask)
        blank_image[h2 - croph:,w2-cropw :, :] = c
        return blank_image,mask_image, intact_image
    elif patten == 5:
        croph = h1
        cropw = int(w1 * occulde_ratio)
        new_w =  w1+w2-cropw
        new_h =  h1

        intact_image = np.zeros((new_h, new_w, 3), np.uint8)
        mask_image = intact_image.copy()
        mask_image[:, :w2, :] = mask1
        intact_image[:, w2-cropw:, :] = image

        blank_image = np.zeros((new_h, new_w, 3), np.uint8)
        mask_crop = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)
        mask_crop[mask_crop != 0] = 255
        mask_crop = revise_mask(mask_crop)
        mask_crop = mask_crop[:,w2-cropw :]

        mask_for = np.zeros((h1, w1), np.uint8)
        mask_for.fill(255)
        mask_for[:, :cropw] = mask_crop

        mask = np.zeros_like(image)
        for i in range(mask.shape[2]):
            mask[:, :, i] = mask_for
        c = cv2.bitwise_and(image, mask)
        blank_image[:,w2-cropw :, :] = c
        return blank_image,mask_image, intact_image
    elif patten == 6:
        ratio = abs(cmath.sqrt(occulde_ratio))
        croph = int(h1 * ratio)
        cropw = int(w1 * ratio)
        new_w = w1 + w2 - cropw
        new_h = h1+h2-croph

        intact_image = np.zeros((new_h, new_w, 3), np.uint8)
        mask_image = intact_image.copy()
        mask_image[h1-croph:, :w2, :] = mask1
        intact_image[:h1, w2 - cropw:, :] = image

        blank_image = np.zeros((new_h, new_w, 3), np.uint8)
        mask_crop = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)
        mask_crop[mask_crop != 0] = 255
        mask_crop = revise_mask(mask_crop)
        mask_crop = mask_crop[:croph, w2 - cropw:]

        mask_for = np.zeros((h1, w1), np.uint8)
        mask_for.fill(255)
        mask_for[h2-croph:, :cropw] = mask_crop

        mask = np.zeros_like(image)
        for i in range(mask.shape[2]):
            mask[:, :, i] = mask_for
        c = cv2.bitwise_and(image, mask)
        blank_image[:h1,w2-cropw :, :] = c
        return blank_image,mask_image, intact_image
    elif patten == 7:  # 270
        croph = int(h1 * occulde_ratio)
        cropw = w1
        new_w = w1
        new_h = h1+h2-croph
        intact_image = np.zeros((new_h, new_w, 3), np.uint8)
        mask_image = intact_image.copy()
        mask_image[h1-croph:, :, :] = mask1
        intact_image[:h1, :, :] = image

        blank_image = np.zeros((new_h, new_w, 3), np.uint8)
        mask_crop = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)
        mask_crop[mask_crop != 0] = 255
        mask_crop = revise_mask(mask_crop)
        mask_crop = mask_crop[:croph, :]

        mask_for = np.zeros((h1, w1), np.uint8)
        mask_for.fill(255)
        mask_for[h2 - croph:, :] = mask_crop

        mask = np.zeros_like(image)
        for i in range(mask.shape[2]):
            mask[:, :, i] = mask_for
        c = cv2.bitwise_and(image, mask)
        blank_image[:h1, :, :] = c
        return blank_image,mask_image, intact_image
    elif patten == 8:
        ratio = abs(cmath.sqrt(occulde_ratio))
        croph = int(h1 * ratio)
        cropw = int(w1 * ratio)

        new_w = w1+w2-cropw
        new_h = h1+h2-croph

        intact_image = np.zeros((new_h, new_w, 3), np.uint8)
        mask_image = intact_image.copy()
        mask_image[h1-croph:, w1-cropw:, :] = mask1
        intact_image[:h1, :w1, :] = image

        blank_image = np.zeros((new_h, new_w, 3), np.uint8)
        mask_crop = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)
        mask_crop[mask_crop != 0] = 255
        mask_crop = revise_mask(mask_crop)
        mask_crop = mask_crop[:croph, :cropw]

        mask_for = np.zeros((h1, w1), np.uint8)
        mask_for.fill(255)
        mask_for[h2 - croph:,w2-cropw :] = mask_crop

        mask = np.zeros_like(image)
        for i in range(mask.shape[2]):
            mask[:, :, i] = mask_for
        c = cv2.bitwise_and(image, mask)
        blank_image[:h1, :w1, :] = c
        return blank_image,mask_image, intact_image

        # cv2.imshow('1', blank_image)
        # cv2.waitKey()

def Expose_boundary(image_logits,only_transform= True ):
    transform_x = transforms.Compose([
        transforms.Resize((320, 320), Image.ANTIALIAS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.], std=[1.]),
    ])
    image_logits = Image.fromarray(image_logits)
    image_logits = transform_x(image_logits)
    laplacian_kernel = torch.tensor(
        [-1, -1, -1, -1, 8, -1, -1, -1, -1],
        dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False)
    if not only_transform:
      image_logits = image_logits.unsqueeze(0)
      boundary_targets = F.conv2d(image_logits, laplacian_kernel, padding=1)
      boundary_targets = boundary_targets.clamp(min=0)
      boundary_targets[boundary_targets > 0.1] = 1
      boundary_targets[boundary_targets <= 0.1] = 0
      return boundary_targets.squeeze(0)
    else:
      return image_logits

def CropRemove_withnoResize(image, mask, pattern, occulde_ratio, occ_revise = False, Boundary_output = False):
    # image.shape  h,w,c
    image = GrayandBinary(image)
    mask = GrayandBinary(mask)
    if image.shape[0] >= mask.shape[0]:
        occder = image
        occ = mask
    else:
        occder = mask
        occ = image
    h1, w1 = occ.shape
    h2, w2 = occder.shape
    if pattern == 1:
        cropw = int(w1 * occulde_ratio)
        if cropw > w2 * 0.9:
            cropw = int(w2 * 0.9)
        new_w = w1 + w2 - cropw
        new_h = max(h1, h2)                                   #其实之前就限制了h2>h1

        Intact_image = np.zeros((new_h, new_w), np.uint8)
        Occder_image = np.zeros((new_h, new_w), np.uint8)

        initial_y = random.randint(0, h2 - h1 - 2)  if (h2 - h1 - 2 >0) else 0
        Intact_image[initial_y:initial_y + h1, :w1] = occ
        Occder_image[:, w1 - cropw:] = occder

        occ_mask = np.zeros((new_h, new_w), np.bool8)

        occ_mask[Occder_image > 0.] = True
        occ_mask = ~occ_mask

        Occ_image = Intact_image * occ_mask
    elif pattern == 2:
        croph = int(h1 * occulde_ratio)
        cropw = int(w1 * occulde_ratio)

        if cropw >= w2:
            cropw = int(w2 * 0.9)

        new_h = h1 + h2 - croph
        new_w = w1 + w2 - cropw

        Intact_image = np.zeros((new_h, new_w), np.uint8)
        Occder_image = np.zeros((new_h, new_w), np.uint8)

        Intact_image[h2 - croph:, :w1] = occ
        Occder_image[:h2, w1 - cropw:] = occder

        occ_mask = np.zeros((new_h, new_w), np.bool8)
        occ_mask[Occder_image > 0.] = True
        occ_mask = ~occ_mask
        Occ_image = Intact_image * occ_mask
    elif pattern == 3:
        croph = int(h1 * occulde_ratio)

        new_w = max(w1, w2)
        new_h = h1 + h2 - croph

        Intact_image = np.zeros((new_h, new_w), np.uint8)
        Occder_image = np.zeros((new_h, new_w), np.uint8)

        if w1 >= w2:
            Intact_image[h2 - croph:, :] = occ
            if w1 - w2 - 2>0:
             initial_x = random.randint(0, w1 - w2 - 2)
             Occder_image[:h2, initial_x:initial_x + w2] = occder
            else:
              initial_x = 0
              Occder_image[:h2, initial_x:initial_x + w2] = occder
        else:
            Occder_image[:h2, :] = occder
            if  w2 - w1 - 2>0:
             initial_x = random.randint(0, w2 - w1 - 2)
             Intact_image[h2 - croph:, initial_x:initial_x + w1] = occ
            else:
             initial_x = 0
             Intact_image[h2 - croph:, initial_x:initial_x + w1] = occ

        occ_mask = np.zeros((new_h, new_w), np.bool8)
        occ_mask[Occder_image > 0.] = True
        occ_mask = ~occ_mask
        Occ_image = Intact_image * occ_mask
    elif pattern == 4:
        croph = int(h1 * occulde_ratio)
        cropw = int(w1 * occulde_ratio)

        if cropw >= w2:
            cropw = int(w2 * 0.9)

        new_h = h1 + h2 - croph
        new_w = w1 + w2 - cropw

        Intact_image = np.zeros((new_h, new_w), np.uint8)
        Occder_image = np.zeros((new_h, new_w), np.uint8)

        Intact_image[h2 - croph:, w2 - cropw:] = occ
        Occder_image[:h2, :w2] = occder

        occ_mask = np.zeros((new_h, new_w), np.bool8)
        occ_mask[Occder_image > 0.] = True
        occ_mask = ~occ_mask
        Occ_image = Intact_image * occ_mask
    elif pattern == 5:
        cropw = int(w1 * occulde_ratio)
        if cropw > w2 * 0.9:
            cropw = int(w2 * 0.9)
        new_w = w1 + w2 - cropw
        new_h = max(h1, h2)

        Intact_image = np.zeros((new_h, new_w), np.uint8)
        Occder_image = np.zeros((new_h, new_w), np.uint8)

        initial_y = random.randint(0, h2 - h1 - 2) if h2 - h1 - 2>0 else 0
        Intact_image[initial_y:initial_y + h1, w2 - cropw:] = occ
        Occder_image[:, :w2] = occder

        occ_mask = np.zeros((new_h, new_w), np.bool8)

        occ_mask[Occder_image > 0.] = True
        occ_mask = ~occ_mask

        Occ_image = Intact_image * occ_mask
    elif pattern == 6:
        croph = int(h1 * occulde_ratio)
        cropw = int(w1 * occulde_ratio)

        if cropw >= w2:
            cropw = int(w2 * 0.9)

        new_h = h1 + h2 - croph
        new_w = w1 + w2 - cropw

        Intact_image = np.zeros((new_h, new_w), np.uint8)
        Occder_image = np.zeros((new_h, new_w), np.uint8)

        Intact_image[:h1, w2 - cropw:] = occ
        Occder_image[h1 - croph:, :w2] = occder

        occ_mask = np.zeros((new_h, new_w), np.bool8)
        occ_mask[Occder_image > 0.] = True
        occ_mask = ~occ_mask
        Occ_image = Intact_image * occ_mask
    elif pattern == 7:
        croph = int(h1 * occulde_ratio)

        new_w = max(w1, w2)
        new_h = h1 + h2 - croph

        Intact_image = np.zeros((new_h, new_w), np.uint8)
        Occder_image = np.zeros((new_h, new_w), np.uint8)

        if w1 >= w2:
            Intact_image[:h1, :w1] = occ
            if w1 - w2 - 2 >0:
              initial_x = random.randint(0, w1 - w2 - 2)
              Occder_image[h1 - croph:, initial_x:initial_x + w2] = occder
            else:
               initial_x =0
               Occder_image[h1 - croph:, initial_x:initial_x + w2] = occder
        else:
            Occder_image[h1 - croph:, :] = occder
            if  w2 - w1 - 2>0:
             initial_x = random.randint(0, w2 - w1 - 2)
             Intact_image[:h1, initial_x:initial_x + w1] = occ
            else:
              initial_x = 0
              Intact_image[:h1, initial_x:initial_x + w1] = occ

        occ_mask = np.zeros((new_h, new_w), np.bool8)
        occ_mask[Occder_image > 0.] = True
        occ_mask = ~occ_mask
        Occ_image = Intact_image * occ_mask
    elif pattern == 8:
        croph = int(h1 * occulde_ratio)
        cropw = int(w1 * occulde_ratio)

        if cropw >= w2:
            cropw = int(w2 * 0.9)

        new_h = h1 + h2 - croph
        new_w = w1 + w2 - cropw

        Intact_image = np.zeros((new_h, new_w), np.uint8)
        Occder_image = np.zeros((new_h, new_w), np.uint8)

        Intact_image[:h1, :w1] = occ
        Occder_image[h1 - croph:, w1 - cropw:] = occder

        occ_mask = np.zeros((new_h, new_w), np.bool8)
        occ_mask[Occder_image > 0.] = True
        occ_mask = ~occ_mask
        Occ_image = Intact_image * occ_mask

    if occ_revise:
        if Boundary_output:
           return [Expose_boundary(Occder_image,only_transform= True),Expose_boundary(Occder_image,only_transform= False)],\
                  [Expose_boundary(Occ_image,only_transform= True),Expose_boundary(Occ_image,only_transform= False)],\
                  [Expose_boundary(Occder_image,only_transform= True),Expose_boundary(Occder_image,only_transform= False)], \
                  [Expose_boundary(Intact_image, only_transform=True), Expose_boundary(Intact_image, only_transform=False)]
        else:
           return  Expose_boundary(Occder_image,only_transform=True), \
                   Expose_boundary(Occ_image,only_transform=True), \
                   Expose_boundary(Occder_image,only_transform=True),\
                   Expose_boundary(Intact_image, only_transform=True)             #反转输出的话需要额外输出完整图像
    else:
        if Boundary_output:
            return [Expose_boundary(Intact_image, only_transform=True),Expose_boundary(Intact_image, only_transform=False)], \
                   [Expose_boundary(Occder_image, only_transform=True), Expose_boundary(Occder_image, only_transform=False)], \
                   [Expose_boundary(Occ_image, only_transform=True),Expose_boundary(Occ_image, only_transform=False)]
        else:
            return Expose_boundary(Intact_image, only_transform=True), \
                   Expose_boundary(Occder_image, only_transform=True), \
                   Expose_boundary(Occ_image, only_transform=True)

def boundary_logits_form_image(x_boundary):
    x_boundary = x_boundary[:, :, 1]
    x_boundary[x_boundary != 0] = 255
    return x_boundary

def GrayandBinary(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image[image != 0] = 255.
    return image

def rotation_image(image,angle):
    image = Image.fromarray(image)
    image = transforms.functional.rotate(image, angle=angle, expand=True, fill=0)
    image = np.array(image)
    return image

def readimage_copyto_boundary(image_path,mask_path):
    mask_1 = cv2.imread(mask_path)
    x = cv2.imread(image_path)
    x1 = x.copy()               # 拷贝一个图片不做任何处理
    angle = transforms.RandomRotation.get_params([-20, 20])
    x= rotation_image(x,angle)
    x1 = rotation_image(x1,angle)
    angle_mask = transforms.RandomRotation.get_params([-20, 20])
    mask_1 = rotation_image(mask_1,angle_mask)
    return x,mask_1, x1

def read_image_with_rotation(image_path,mask_path):
    mask_1 = cv2.imread(mask_path)
    x = cv2.imread(image_path)

    angle = transforms.RandomRotation.get_params([-5, 5])
    x= rotation_image(x,angle)

    angle_mask = transforms.RandomRotation.get_params([-5, 5])
    mask_1 = rotation_image(mask_1, angle_mask)
    return x,mask_1

class Grap_paritialcomp_dataset_newest(Dataset):
    def __init__(self,
                 dataset_path='../data/mvtec_anomaly_detection',
                 class_name='bottle',
                 is_train=True,
                 resize=256,
                 boundary_output = False,
                 ):
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.x = self.load_dataset_folder()
        self.boundary_output = boundary_output

    def __len__(self):
            return len(self.x)

    def load_dataset_folder(self):
            phase = 'train' if self.is_train else 'test'
            x = []
            img_dir = os.path.join(self.dataset_path, self.class_name, phase)
            img_types = sorted(os.listdir(img_dir))
            for img_type in img_types:
                # load images
                img_type_dir = os.path.join(img_dir, img_type)
                if not os.path.isdir(img_type_dir):
                    print(1)
                    continue
                img_fpath_list = sorted(
                    [os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir) if f.endswith('.png')])
                x.extend(img_fpath_list)
            return list(x)

    def adjust_imagepair_size(self,idx_1,idx_2):
        a = cv2.imread(self.x[idx_1],0)
        b = cv2.imread(self.x[idx_2],0)

        size_a = a.shape[0]* a.shape[1]
        size_b = b.shape[0]* b.shape[1]

        size_max = max(size_a,size_b)
        size_min = min(size_a, size_b)


        return  (size_max/size_min) >= 2.25

    def tensor_clamp(self,tensor_in):
        if torch.is_tensor(tensor_in):
            tensor_in[tensor_in > 0.1] = 1.
            tensor_in[tensor_in <= 0.1] = 0.
        else:
            if isinstance(tensor_in, list):
                for i in range(len(tensor_in)):
                    self.tensor_clamp(tensor_in[i])
            else:
                assert ("数据类型错误，应该是tensor 或者list")
        return tensor_in

    def caculate_the_occ_ratio(self,Intact,Occ):
        Occ_area = Intact- Occ
        return ((Occ_area>0.).sum()/(Intact>0.).sum())

    def __getitem__(self, idx):
            phase = 'train' if self.is_train else 'test'
            # self.x 中表示图片的地址
            if self.is_train:
                    mask_idx = random.randint(0, len(self.x) - 1)
                    # 判断两个图像是否尺寸差异过大
                    while  self.adjust_imagepair_size(idx,mask_idx):
                        mask_idx = random.randint(0, len(self.x) - 1)
                    else:
                        mask_idx = mask_idx

                    # 读取图像和mask图像 并将二者进行旋转
                    image, mask_for_image = read_image_with_rotation(self.x[idx], self.x[mask_idx])
                    a = random.randint(1, 8)
                    m = random.uniform(0.2, 0.7)
                    probability = random.randint(1,10)
                    occ_revise = (probability>=9)

                    if occ_revise:
                        Intact_image, Occder_image, Occ_image,Occder_intact_image = CropRemove_withnoResize(image, mask_for_image, a, m, occ_revise = occ_revise,
                                                                                   Boundary_output = self.boundary_output)

                        Occder_intact_image = self.tensor_clamp(Occder_intact_image)
                    else:
                        Intact_image, Occder_image, Occ_image = CropRemove_withnoResize(image, mask_for_image, a, m,
                                                                                        occ_revise=occ_revise,
                                                                                        Boundary_output=self.boundary_output)
                    Intact_image = self.tensor_clamp(Intact_image)
                    Occder_image = self.tensor_clamp(Occder_image)
                    Occ_image = self.tensor_clamp(Occ_image)

                    # 调整重合度的大小，防止出现较小的重合度     其中如果反向后的输出则不需要调整
                    #今天这里调整了 调整了重叠的上限和下限
                    if not occ_revise:
                      #查询是否反转输出是因为在反转输出情况下 重合率为0]
                      if self.boundary_output:
                          # 当boundary_out的情况下，输出的数据个数为list格式
                          occ_ratio = self.caculate_the_occ_ratio(Intact_image[0], Occ_image[0])
                          while occ_ratio < 0.2 or occ_ratio > 0.75:
                              if occ_ratio < 0.2:
                                  m = m + 0.05
                                  if m > 0.8:
                                      break
                              elif occ_ratio > 0.75:
                                  m = m - 0.05
                                  if m < 0.1:
                                      break
                              Intact_image, Occder_image, Occ_image = CropRemove_withnoResize(image, mask_for_image, a,
                                                                                              m,
                                                                                              occ_revise=occ_revise,
                                                                                              Boundary_output=self.boundary_output)
                              Intact_image = self.tensor_clamp(Intact_image)
                              Occder_image = self.tensor_clamp(Occder_image)
                              Occ_image = self.tensor_clamp(Occ_image)
                              # 更新重合度参数
                              occ_ratio = self.caculate_the_occ_ratio(Intact_image[0], Occ_image[0])
                      else:
                          occ_ratio = self.caculate_the_occ_ratio(Intact_image, Occ_image)
                          while occ_ratio < 0.2 or occ_ratio > 0.75:
                              if occ_ratio < 0.2:
                                  m = m + 0.05
                                  if m > 0.8:
                                      break
                              elif occ_ratio > 0.75:
                                  m = m - 0.05
                                  if m < 0.1:
                                      break
                              Intact_image, Occder_image, Occ_image = CropRemove_withnoResize(image, mask_for_image, a,
                                                                                              m,
                                                                                              occ_revise=occ_revise,
                                                                                              Boundary_output=self.boundary_output)
                              Intact_image = self.tensor_clamp(Intact_image)
                              Occder_image = self.tensor_clamp(Occder_image)
                              Occ_image = self.tensor_clamp(Occ_image)
                              # 更新重合度参数
                              occ_ratio = self.caculate_the_occ_ratio(Intact_image, Occ_image)
                    else:
                      if self.boundary_output:
                         raise('此处的代码还有待完善')
                      else:
                        #反转输出的情况下，计算occder与occder_intact的重合度
                        occ_ratio = self.caculate_the_occ_ratio(Occder_intact_image, Occder_image)
                        while occ_ratio < 0.2 or occ_ratio > 0.75:
                            if occ_ratio < 0.2:
                                m = m + 0.05
                                if m > 0.8:
                                    break
                            elif occ_ratio > 0.75:
                                m = m - 0.05
                                if m < 0.1:
                                    break
                            Intact_image, Occder_image, Occ_image,Occder_intact_image = CropRemove_withnoResize(image, mask_for_image, a,
                                                                                        m,
                                                                                        occ_revise=occ_revise,
                                                                                        Boundary_output=self.boundary_output)
                            Intact_image = self.tensor_clamp(Intact_image)
                            Occder_image = self.tensor_clamp(Occder_image)
                            Occ_image = self.tensor_clamp(Occ_image)
                            Occder_intact_image = self.tensor_clamp(Occder_intact_image)
                            # 更新重合度参数
                            occ_ratio = self.caculate_the_occ_ratio(Occder_intact_image, Occder_image)

                    if self.boundary_output:
                       if isinstance(Intact_image,list):                    #如果采用boudary_output模式 则函数的输出为三个list
                          return [torch.cat(Intact_image,dim=0),torch.cat(Occder_image,dim=0),torch.cat(Occ_image,dim=0)]
                    else:
                       return [Intact_image, Occder_image, Occ_image]
            else:
                print('该dataloader不支持验证集读取')


class Grap_paritialcomp_dataset(Dataset):
    def __init__(self,
                 dataset_path='../data/mvtec_anomaly_detection',
                 class_name='bottle',
                 is_train=True,
                 for_maskdeOcc=False,
                 resize=256,
                 ):
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.for_maskdeOcc = for_maskdeOcc
        self.resize = resize
        self.x = self.load_dataset_folder()
        # set transforms
        self.transform_x = transforms.Compose([
            transforms.Resize(resize, Image.ANTIALIAS),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.],std=[1.])
        ])
    def __getitem__(self, idx):
        phase = 'train' if self.is_train else 'test'
        # self.x 中表示图片的地址
        if self.is_train:
            if not self.for_maskdeOcc:
                # 读取一个batch中的图像，并取出多张图像作
                mask_idx = random.randint(0, len(self.x) - 1)
                # 读取图像和mask图像 并将二者进行旋转
                image, mask_for_image = read_image_with_rotation(self.x[idx], self.x[mask_idx])
                # 选取遮挡模式和重合度大小
                a = random.randint(1, 8)
                m = random.uniform(0.3, 0.7)

                Occ_image, Intact_image = CropRemove_WithOriginal(image, mask_for_image, a, m)
                Occ_image = Image.fromarray(Occ_image)
                Occ_image = self.transform_x(Occ_image)

                # 拷贝一张图像用于提取边界
                x_boundary = Intact_image.copy()

                Intact_image = Image.fromarray(Intact_image)
                Intact_image = self.transform_x(Intact_image)

                x_boundary = boundary_logits_form_image(x_boundary)
                x_boundary = Image.fromarray(x_boundary)
                x_boundary = self.transform_for_boundary(x_boundary)
                return Occ_image, Intact_image, x_boundary
            else:
                mask_idx = random.randint(0, len(self.x) - 1)
                # 读取图像和mask图像 并将二者进行旋转
                image, mask_for_image = read_image_with_rotation(self.x[idx], self.x[mask_idx])
                # 选取遮挡模式和重合度大小
                a = random.randint(1, 8)
                m = random.uniform(0.4, 0.7)
                Occ_image,Occed_image, Intact_image = CropRemove_WithOriginal(image, mask_for_image, a, m)

                Occ_image = GrayandBinary(Occ_image)
                Occed_image = GrayandBinary(Occed_image)
                Intact_image = GrayandBinary(Intact_image)

                Occ_image = Image.fromarray(Occ_image)
                Occ_image = self.transform_x(Occ_image)

                Occed_image = Image.fromarray(Occed_image)
                Occed_image = self.transform_x(Occed_image)

                Intact_image = Image.fromarray(Intact_image)
                Intact_image = self.transform_x(Intact_image)

                #Intact_image = torch.tensor(Intact_image, dtype=torch.int8).view(320, 320)

                return Occ_image,Occed_image, Intact_image.view(320, 320)
            # ——————————————————————————————————————————————————————————————————————————————————原来的代码 注释掉了————————————————————————————————————————————————————#
            # mask_idx = random.randint(0,len(self.x)-1)
            # image,mask_for_image,image_without_process = readimage_copyto_boundary(self.x[idx],self.x[mask_idx])  #读取图像并对图像进行相应的旋转
            # x_boundary = image_without_process.copy()                           #拷贝一张图像提取边界
            # x_boundary = boundary_logits_form_image(x_boundary)                  #获得边界提取图像
            # #在获取随机遮挡图像前需要将 图像、遮挡图像、边界图都要进行旋转
            # image= gen_mask1(image,mask_for_image)                              #获取随机遮挡的图像
            # image= Image.fromarray(image).convert('RGB')
            # image = self.transform_x(image)
            # image_without_process = Image.fromarray(image_without_process).convert('RGB')
            # image_without_process = self.transform_x(image_without_process)
            # x_boundary =Image.fromarray(x_boundary)
            # x_boundary = self.transform_for_boundary(x_boundary)
            #
            # from torchvision import utils as vutils
            # vutils.save_image(image, './x.jpg', normalize=True)
            # vutils.save_image(image_without_process, './x1.jpg', normalize=True)
            # vutils.save_image(x_boundary, './x_boundary.jpg', normalize=True)
            # return image, image_without_process, x_boundary
            # ——————————————————————————————————————————————————————————————————————————————————原来的代码 注释掉了————————————————————————————————————————————————————#
        else:
            print('该dataloader没有采用val模式')
    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x= []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:
            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                print(1)
                continue
            img_fpath_list = sorted(
                [os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir) if f.endswith('.png')])
            x.extend(img_fpath_list)
        return list(x)

class Grap_paritialcomp_dataset_forval(Dataset):
    def __init__(self,
                 dataset_path='../data/mvtec_anomaly_detection',
                 resize=256,
                 ):
        self.dataset_path = dataset_path
        self.resize = resize
        self.image_pair = self.load_dataset_folder(self.dataset_path)
        self.transform_x = transforms.Compose([
            transforms.Resize(resize, Image.ANTIALIAS),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.],std=[1.])
        ])

    def tensor_clamp(self, tensor_in):
            if torch.is_tensor(tensor_in):
                tensor_in[tensor_in > 0.1] = 1.
                tensor_in[tensor_in <= 0.1] = 0.
            else:
                if isinstance(tensor_in, list):
                    for i in range(len(tensor_in)):
                        self.tensor_clamp(tensor_in[i])
                else:
                    assert ("数据类型错误，应该是tensor 或者list")
            return tensor_in

    def __getitem__(self, idx):
        # self.x 中表示图片的地址
        image_path1 = self.image_pair[idx][0]
        image_path2 = self.image_pair[idx][1]
        image = cv2.imread(image_path1)
        image2 = cv2.imread(image_path2)
        if image.shape != image2.shape:
            print(image_path1)
            print(image_path2)
            raise Exception("读取的两张图不能够组成图像对")
        else:
            image = GrayandBinary(image)
            image2 = GrayandBinary(image2)

            image = Image.fromarray(image)
            image = self.transform_x(image)

            image2 = Image.fromarray(image2)
            image2 = self.transform_x(image2)

            image = self.tensor_clamp(image)
            image2 = self.tensor_clamp(image2)

            return image, image2,
    def __len__(self):
        return len(self.image_pair)

    def load_dataset_folder(self,dir_path):
        image_list_pair =[]
        image_name_list=[]
        for files in os.listdir(self.dataset_path):
            image_name_list.append(files)
            image_name_list.sort()
        for i in range(len(image_name_list) - 1):
            if i % 2 == 0:
                image_list_pair.append([dir_path + image_name_list[i], dir_path + image_name_list[i + 1]])
        return image_list_pair



