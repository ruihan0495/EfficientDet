import cv2
import json
import numpy as np
import os
import time
import glob

from model import EfficientDetModel
from utils import preprocess_image, postprocess_boxes
from utils.draw_boxes import draw_boxes



def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    phi = 0
    weighted_bifpn = False
    model_path = 'checkpoints/deepfashion.h5'
    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    image_size = image_sizes[phi]
    # coco classes
    classes = {value['id'] - 1: value['name'] for value in json.load(open('deepfashion_13.json', 'r')).values()}
    num_classes = 13
    score_threshold = 0.3
    colors = [np.random.randint(0, 256, 3).tolist() for _ in range(num_classes)]
    #_, model = efficientdet(phi=phi,
    #                        weighted_bifpn=weighted_bifpn,
    #                        num_classes=num_classes,
    #                        score_threshold=score_threshold)
    models = EfficientDetModel(0)
    model = models.p_model
    model.load_weights(model_path, by_name=True)

    # 'datasets/VOC2007/JPEGImages/*.jpg'
    for image_path in glob.glob('data/sample_val/image/000002.jpg'):
        image = cv2.imread(image_path)
        src_image = image.copy()
        # BGR -> RGB
        image = image[:, :, ::-1]
        h, w = image.shape[:2]

        image, scale = preprocess_image(image, image_size=image_size)
        # run network
        start = time.time()
        boxes, scores, labels, masks = model.predict_on_batch([np.expand_dims(image, axis=0)])
        boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)

        print(time.time() - start)
        boxes = postprocess_boxes(boxes=boxes, scale=scale, height=h, width=w)

        # select indices which have a score above the threshold
        indices = np.where(scores[:] > score_threshold)[0]

        # select those detections
        boxes = boxes[indices]
        labels = labels[indices]

        draw_boxes(src_image, boxes, scores, labels, colors, classes)

        #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imwrite('results/image.jpg', src_image)
        #cv2.imread('results/image.jpg')
        '''
        for j,mask in enumerate(masks):
            for i, m in enumerate(mask):
                for k in range(13):
                    n = np.round(m[:,:,k])
                    if np.any(n) > 0:
                        print("here!")
                        print(n.shape)
                        cv2.imwrite('results/mask_{}_{}.jpg'.format(i, k), n*255)'''
        #cv2.imshow('image', src_image)
        #cv2.waitKey(0)


if __name__ == '__main__':
    main()
