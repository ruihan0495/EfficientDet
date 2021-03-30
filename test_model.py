from model import EfficientDetModel
from utils import preprocess_image, postprocess_boxes
from utils.draw_boxes import draw_boxes
import tensorflow as tf
import cv2
import json
import numpy as np
import os
import time
import glob
from model import build_mask_target_graph, data_generator
from generators.coco_ins import CocoDataset
from augmentor.mask_misc import MiscEffectMask
from utils.graph_funcs import trim_zeros_graph, overlaps_graph, norm_boxes_graph, denorm_boxes_graph
from tensorflow.keras import backend as K

tf.enable_eager_execution()
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
    misc_effect = MiscEffectMask()
    train_dataset = CocoDataset('data/sample_val', ['train', 'val'], misc_effect=misc_effect)
    train_generator = data_generator(train_dataset, shuffle=True,
                                         phi=0, batch_size=1)

    for images, _, _, masks_batch, boxes_batch, labels_batch in next(train_generator):
        image = images[0]
        src_image = image.copy()
        cv2.imwrite('results/image.jpg', src_image*255)
        d_image = image.copy()
        #image = image[:, :, ::-1]
        h, w = image.shape[:2]
        
        #image, scale = preprocess_image(image, image_size=image_size)
        # run network
        start = time.time()
        boxes, scores, labels, _ = model.predict_on_batch([np.expand_dims(image, axis=0), masks_batch, boxes_batch, labels_batch])
        boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)
        # testing....
        boxes = norm_boxes_graph(boxes, (512, 512))
        print(boxes.shape, boxes_batch[0].shape, masks_batch[0].shape, labels_batch[0].shape)
        rois, roi_gt_class_ids, target_mask = build_mask_target_graph(boxes, boxes_batch[0], 
                                                                      masks_batch[0], labels_batch[0])
        target_mask = target_mask.numpy()
        '''
        positive_overlap = 0.4
        negative_overlap = 0.3
        b_gt_boxes = boxes_batch[0]
        b_gt_masks = masks_batch[0]
        gt_class_ids = labels_batch[0]
        num_classes=13
        mask_shape=[28,28]
        TRAIN_ROIS_PER_IMAGE = 100
       
        boxes = norm_boxes_graph(boxes, (512, 512))
        rois, _ = trim_zeros_graph(boxes, name="trim_rois")
        #print(rois)
        b_gt_boxes, non_zeros = trim_zeros_graph(b_gt_boxes, name='trim_gt_boxes')
        b_gt_masks = tf.gather(b_gt_masks, tf.where(non_zeros)[:, 0], axis=0,
                            name="trim_gt_masks")   # shape [num_instances, img_size, img_size]
        #print(b_gt_boxes, non_zeros)
        #print(b_gt_masks.shape)
        # skip handle coco crowd for now
        gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
                                    name="trim_gt_class_ids")
        overlaps = overlaps_graph(rois, b_gt_boxes)
        #print(overlaps)
        # Determine positive and negative ROIs
        roi_iou_max = tf.reduce_max(overlaps, axis=1)
        # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
        positive_roi_bool = (roi_iou_max >= positive_overlap)
        positive_indices = tf.where(positive_roi_bool)[:, 0]
        # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
        negative_indices = tf.where((roi_iou_max < negative_overlap))[:, 0]

        positive_count = int(TRAIN_ROIS_PER_IMAGE * 0.8)
        positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
        positive_count = tf.shape(positive_indices)[0]
        # Negative ROIs. Add enough to maintain positive:negative ratio.
        r = 1.0 / 0.8
        negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
        negative_indices = tf.random_shuffle(negative_indices)[:negative_count]

        positive_overlaps = tf.gather(overlaps, positive_indices)
        roi_gt_box_assignment = tf.cond(
            tf.greater(tf.shape(positive_overlaps)[1], 0),
            true_fn = lambda: tf.argmax(positive_overlaps, axis=1),
            false_fn = lambda: tf.cast(tf.constant([]),tf.int64)
        )
        #print(positive_indices)
        #print(roi_gt_box_assignment)
        # Permute masks to [N, height, width, 1]
        transposed_masks = tf.expand_dims(b_gt_masks, -1)
        # Pick the right mask for each ROI
        roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)
        roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)
        boxes = tf.gather(rois, positive_indices)
        box_ids = tf.range(0, tf.shape(roi_masks)[0])
        b_masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes,
                                        box_ids,
                                        mask_shape)
        b_masks = tf.round(tf.squeeze(b_masks, axis=-1))
        positive_rois = tf.gather(rois, positive_indices)
        negative_rois = tf.gather(rois, negative_indices)
        # Append negative ROIs and pad bbox deltas and masks that
        # are not used for negative ROIs with zeros.
        N = tf.shape(negative_rois)[0]
        P = tf.maximum(TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
        rois = tf.concat([positive_rois, negative_rois], axis=0)
        rois = tf.pad(rois, [(0, P), (0, 0)])
        roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])   
        target_mask = tf.pad(b_masks, [[0, N + P], (0, 0), (0, 0)])
        target_mask = target_mask.numpy()
        '''
        rois = denorm_boxes_graph(rois, (512, 512))
        print("target mask shape", target_mask.shape)
        print(roi_gt_class_ids)
        for box in rois:
            xmin, ymin, xmax, ymax = list(map(int, box))
            print(box)
            cv2.rectangle(d_image, (xmin, ymin), (xmax, ymax), colors[0], 1)
            #cv2.rectangle(target_mask[0], (xmin, ymin), (xmax, ymax), colors[0], 1)
            cv2.rectangle(masks_batch[0][0], (xmin, ymin), (xmax, ymax), colors[0], 1)
            cv2.rectangle(masks_batch[0][1], (xmin, ymin), (xmax, ymax), colors[0], 1)
        cv2.imwrite('results/detect_img.jpg', d_image*255)
        for i in range(target_mask.shape[0]):
            cv2.imwrite('results/preds/tmask_img_{}.jpg'.format(i), target_mask[i]*255)
        cv2.imwrite('results/mask_img1.jpg', masks_batch[0][0]*255)
        cv2.imwrite('results/mask_img2.jpg', masks_batch[0][1]*255)
        #print(target_mask.shape, target_mask)
        #cv2.imwrite('results/mask_img2.jpg', masks[1])
        #for i in range(10):
            #cv2.imwrite('results/dmask_img_{}.jpg', tf.round(target_mask[i]) *255)
        #boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)
        #print(time.time() - start)
        #boxes = postprocess_boxes(boxes=boxes, scale=scale, height=h, width=w)

        # select indices which have a score above the threshold
        #indices = np.where(scores[:] > score_threshold)[0]

        # select those detections
        #boxes = boxes[indices]
        #labels = labels[indices]

        #draw_boxes(src_image, boxes, scores, labels, colors, classes)

        #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        #cv2.imwrite('results/image.jpg', src_image)
        break


if __name__ == '__main__':
    main()
