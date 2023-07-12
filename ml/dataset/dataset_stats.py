from pycocotools.coco import COCO
import numpy as np


coco = COCO('./dataset/coco/annotations/instances_default.json')

object_categories = [category['name'] for category in coco.loadCats(coco.getCatIds())]

object_counts = {category: 0 for category in object_categories}
object_sizes = {category: [] for category in object_categories}
object_rel_sizes = {category: [] for category in object_categories}

for ann_id in coco.getAnnIds():
    ann = coco.loadAnns(ann_id)[0]
    category = coco.loadCats(ann['category_id'])[0]['name']
    
    img = coco.loadImgs(ann['image_id'])[0]
    
    h, w = img['height'], img['width']
    
    object_counts[category] += 1
    object_sizes[category].append((ann['bbox'][2], ann['bbox'][3]))
    object_rel_sizes[category].append((ann['bbox'][2]/w, ann['bbox'][3]/h))

print(object_counts)
print({category: (np.mean(sizes[0]), np.mean(sizes[1])) for category, sizes in object_sizes.items()})
print({category: (np.mean(sizes[0]), np.mean(sizes[1])) for category, sizes in object_rel_sizes.items()})
