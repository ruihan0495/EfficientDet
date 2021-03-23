from generators.coco_ins import CocoDataset
from model import data_generator
from augmentor.mask_misc import MiscEffectMask, multi_scale, translate, rotate, translate, flipx, crop
import cv2
import numpy as np


misc_effect = MiscEffectMask(1,1,1,1,1)
train_generator = CocoDataset('data/sample_val', ['train', 'val'], misc_effect=misc_effect)
#train_generator = data_generator(train_dataset, shuffle=True,
#                                        phi=0, batch_size=1)

for i in range(train_generator.size()):
    image = train_generator.load_image(i)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    annotations = train_generator.load_annotations(i)
    boxes = annotations['bboxes'].astype(np.int32)
    masks = annotations['masks'].astype(np.int32)
    #quadrangles = annotations['quadrangles'].astype(np.int32)
    for box in boxes:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)
    #cv2.drawContours(image, quadrangles, -1, (0, 255, 255), 1)
    for i,mask in enumerate(masks):
        cv2.imwrite('test_effects/load_mask_{}.jpg'.format(i), mask*255)
    
    src_image = image.copy()
    # cv2.namedWindow('src_image', cv2.WINDOW_NORMAL)
    #cv2.imshow('src_image', src_image)
    image, annotations = misc_effect(image, annotations)
    #image, annotations = multi_scale(image, annotations, prob=1.)
    #image = image.copy()
    #boxes = annotations['bboxes'].astype(np.int32)
    #masks = annotations['masks'].astype(np.int32)
    #quadrangles = annotations['quadrangles'].astype(np.int32)
    #for box in boxes:
    #    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
    #cv2.drawContours(image, quadrangles, -1, (255, 255, 0), 1)
    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #cv2.imshow('image', image)
    for i,mask in enumerate(masks):
        cv2.imwrite('test_effects/scale_mask_{}.jpg'.format(i), mask*255)

    break