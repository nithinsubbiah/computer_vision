import visdom
import numpy as np
import cv2

import _init_paths
from datasets.factory import get_imdb

vis = visdom.Visdom(server='ec2-18-218-85-198.us-east-2.compute.amazonaws.com',port='8097')

imdb = get_imdb('voc_2007_trainval')

idx_list = imdb._load_image_set_index()

id_num = 2020
Id = idx_list[id_num]

annotation_data = imdb._load_pascal_annotation(Id)
#print(annotation_data['gt_classes'])

f_name_id = imdb.image_path_from_index(imdb.image_index[id_num])
#print(f_name_id)

img = cv2.imread(f_name_id)
img = img.astype(np.uint8)
roidb = imdb.selective_search_roidb()
gtdb = imdb.gt_roidb()
#boxes = roidb[id_num]['boxes']
boxes = gtdb[id_num]['boxes']
for i in range(boxes.shape[0]):
	x1, y1, x2, y2 = boxes[i]
	cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0),1)

img = np.rollaxis(img,-1,0)
vis.image(img)
#vis.text('Hello, Nithin!')
#vis.image(np.ones((3,10,10)))
