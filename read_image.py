import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from decimal import *
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage import io
import pylab

pylab.rcParams['figure.figsize'] = (8.0, 10.0)

if __name__ == "__main__":
    # 定义Coco数据集根目录
    coco_root = r"F:/COCO数据集/occ/"

    coco_data = ['JPEGImages']

    # 定义需要提取的类别
    labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
              'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
              'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
              'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
              'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
              'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
              'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
              'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
              'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
              'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
              'toothbrush']

    KINS_LABEL = ['cyclist','pedestrian','car','tram','trcuk','van','misc']
    labels = ['person']

    # dataDir = 'F:/COCO数据集/occ'
    # annFile = '{}/instances_occ2017_new.json'.format(dataDir)
    dataDir = 'G:/AppAndData/KINS数据集/Amodal_Instance/'
    annFile = '{}/instances_val.json'.format(dataDir)
    coco = COCO(annFile)
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))

    nms = set([cat['supercategory'] for cat in cats])
    print('COCO supercategories: \n{}'.format(' '.join(nms)))


    catIds = coco.getCatIds(catNms=['car'])
    imgIds = coco.getImgIds(catIds=catIds)

    imageDir = 'G:/AppAndData/KINS数据集/data_object_image_2/testing/image_2'
    img = coco.loadImgs(imgIds[2])[0]
    I = io.imread(imageDir + '/JPEGImages/' + img['file_name'])
    plt.imshow(I)
    plt.axis('off')
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)
    plt.show()