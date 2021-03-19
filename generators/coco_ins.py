import os
import numpy as np
import cv2
from tensorflow import keras
from .base import BaseDataset

from pycocotools.coco import COCO
from pycocotools import mask as maskUtils


class CocoDataset(BaseDataset):
    """
    Generate data from the COCO dataset.
    See https://github.com/cocodataset/cocoapi/tree/master/PythonAPI for more information.
    """

    def __init__(self, data_dir, set_name, **kwargs):
        """
        Initialize a COCO data generator.

        Args
            data_dir: Path to where the COCO dataset is stored.
            set_name: Name of the set to parse.
        """
        self.data_dir = data_dir
        self.set_name = set_name
        '''
        if set_name in ['train2017', 'val2017']:
            self.coco = COCO(os.path.join(data_dir, 'annotations', 'instances_' + set_name + '.json'))
        else:
            self.coco = COCO(os.path.join(data_dir, 'annotations', 'image_info_' + set_name + '.json'))
        '''
        # For DeepFashin2Dataset sample_val
        self.coco = COCO(os.path.join(data_dir, 'deepfashion2.json')) 
        self.image_ids = self.coco.getImgIds()
        self.image_sizes = {}

        self.load_classes()

        super(CocoDataset, self).__init__(**kwargs)

    def load_classes(self):
        """
        Loads the class to label mapping (and inverse) for COCO.
        """
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def size(self):
        """ Size of the COCO dataset.
        """
        return len(self.image_ids)

    def num_classes(self):
        """ Number of classes in the dataset. For COCO this is 80.
        """
        return 13

    def has_label(self, label):
        """ Return True if label is a known label.
        """
        return label in self.labels

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        return name in self.classes

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.labels[label]

    def coco_label_to_label(self, coco_label):
        """ Map COCO label to the label as used in the network.
        COCO has some gaps in the order of labels. The highest label is 90, but there are 80 classes.
        """
        return self.coco_labels_inverse[coco_label]

    def coco_label_to_name(self, coco_label):
        """ Map COCO label to name.
        """
        return self.label_to_name(self.coco_label_to_label(coco_label))

    def label_to_coco_label(self, label):
        """ Map label as used by the network to labels as used by COCO.
        """
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

    def load_image(self, image_index):
        """
        Load an image at the image_index.
        """
        # {'license': 2, 'file_name': '000000259765.jpg', 'coco_url': 'http://images.cocodataset.org/test2017/000000259765.jpg', 'height': 480, 'width': 640, 'date_captured': '2013-11-21 04:02:31', 'id': 259765}
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        self.image_sizes[self.image_ids[image_index]] = (image_info['height'], image_info['width'])
        #path = os.path.join(self.data_dir, 'images', self.set_name, image_info['file_name'])
        path = os.path.join(self.data_dir, 'image', image_info['file_name'])
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        h, w = self.image_sizes[self.image_ids[image_index]]
        annotations = {'labels': np.empty((0,), dtype=np.float32), 'bboxes': np.empty((0, 4), dtype=np.float32),
                       'masks': np.empty((0, h, w), dtype=bool)}

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):
            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue
            annotations['masks'] = np.concatenate([annotations['masks'],[
                self.annToMask(a, h, w)]], axis=0)
            annotations['labels'] = np.concatenate(
                [annotations['labels'], [a['category_id'] - 1]], axis=0)
            annotations['bboxes'] = np.concatenate([annotations['bboxes'], [[
                a['bbox'][0],
                a['bbox'][1],
                a['bbox'][0] + a['bbox'][2],
                a['bbox'][1] + a['bbox'][3],
            ]]], axis=0)

        return annotations
  
    def compute_inputs_targets(self, group):
        """
        Compute inputs and target outputs for the network.
        Reutrns:

        """
        MAX_INSTANCES = 100
        # load images and annotations
        # list
        image_group = self.load_image_group(group)
        annotations_group = self.load_annotations_group(group)

        # check validity of annotations
        image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        # randomly apply visual effect
        image_group, annotations_group = self.random_visual_effect_group(image_group, annotations_group)

        # randomly transform data
        # image_group, annotations_group = self.random_transform_group(image_group, annotations_group)

        # randomly apply misc effect
        image_group, annotations_group = self.random_misc_group(image_group, annotations_group)

        # perform preprocessing steps
        image_group, annotations_group = self.preprocess_group(image_group, annotations_group)

        # check validity of annotations
        image_group, annotations_group = self.clip_transformed_annotations(image_group, annotations_group, group)

        assert len(image_group) != 0
        assert len(image_group) == len(annotations_group)

        if self.detect_quadrangle:
            # compute alphas and ratio for targets
            self.compute_alphas_and_ratios(annotations_group)

        # compute network inputs
        inputs = self.compute_inputs(image_group, annotations_group)

        # compute network targets
        # pad zeros until reach the maximum instances per image
        '''
        classification_targets = [annotations['labels'] for annotations in annotations_group]
        regression_targets = [annotations['bboxes'] for annotations in annotations_group]


        labels_batch = np.zeros((len(classification_targets), MAX_INSTANCES), dtype=np.float32)
        regression_batch = np.zeros((len(regression_targets), MAX_INSTANCES, 4), dtype=np.float32)
        for index, (c_target, r_target) in enumerate(zip(classification_targets, regression_targets)):
            if c_target.shape[0] < MAX_INSTANCES:
                num_pads = MAX_INSTANCES - c_target.shape[0]
                neg_padding = np.zeros((num_pads, ))-1
                c_target = np.concatenate((c_target, neg_padding), axis=None)
                zero_padding = np.zeros((num_pads, 4))
                r_target = np.concatenate((r_target, zero_padding), axis=0)
            labels_batch[index] = c_target
            regression_targets[index] = r_target
        '''
        return inputs, annotations_group


       

