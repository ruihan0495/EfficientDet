from generators.coco import CocoGenerator


generator = CocoGenerator('data/sample_val', 'train_2019')
index = 0
image = generator.load_image(index)
annotation = generator.load_annotations(index)
print(annotation['masks'].shape)
print(annotation['bboxes'].shape)

group = [0, 2, 4, 10]

ins, outs = generator.compute_inputs_targets(group)