import json
import os


class COCO2YOLO:
    def __init__(self, json_file, output):
        self._check_file_and_dir(json_file, output)
        self.labels = json.load(open(json_file, 'r', encoding='utf-8'))
        self.coco_id_name_map = self._categories()
        self.coco_name_list = list(self.coco_id_name_map.values())
        print("total images", len(self.labels['images']))
        print("total categories", len(self.labels['categories']))
        print("total labels", len(self.labels['annotations']))

    def _check_file_and_dir(self, file_path, dir_path):
        if not os.path.exists(file_path):
            raise ValueError("file not found")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def _categories(self):
        categories = {}
        for cls in self.labels['categories']:
            categories[cls['id']] = cls['name']
        return categories

    def _load_images_info(self):
        images_info = {}
        for image in self.labels['images']:
            id = image['id']
            file_name = image['file_name']
            if file_name.find('\\') > -1:
                file_name = file_name[file_name.index('\\')+1:]
            w = image['width']
            h = image['height']
            images_info[id] = (file_name, w, h)

        return images_info
    
    @staticmethod
    def bbox_coco2yolo(box, image_w, image_h):
        x, y, w, h = box
        x = x + w / 2
        y = y + h / 2
        x = x / image_w
        y = y / image_h
        w = w / image_w
        h = h / image_h
        return x, y, w, h
    
    @staticmethod
    def seg_coco2yolo(seg, image_w, image_h):
        if type(seg) == dict:
            return None

        segments = []
        for i in range(0, len(seg[0]), 2):
            x = seg[0][i]
            y = seg[0][i + 1]

            x = x / image_w
            y = y / image_h

            segments.append((x, y))

        return segments

    def _convert_anno(self, images_info):
        anno_dict = dict()
        for anno in self.labels['annotations']:
            bbox = anno['bbox']
            image_id = anno['image_id']
            category_id = anno['category_id']

            image_info = images_info.get(image_id)
            image_name = image_info[0]
            img_w = image_info[1]
            img_h = image_info[2]
            
            yolo = self.bbox_coco2yolo(bbox, img_w, img_h)
            
            if yolo is None:
                continue

            anno_info = (image_name, category_id, yolo)
            anno_infos = anno_dict.get(image_id)
            if not anno_infos:
                anno_dict[image_id] = [anno_info]
            else:
                anno_infos.append(anno_info)
                anno_dict[image_id] = anno_infos

        return anno_dict
    
    def _convert_anno_seg(self, images_info):
        anno_dict = dict()
        for anno in self.labels['annotations']:
            seg = anno['segmentation']
            image_id = anno['image_id']
            category_id = anno['category_id']

            image_info = images_info.get(image_id)
            image_name = image_info[0]
            img_w = image_info[1]
            img_h = image_info[2]
            
            yolo = self.seg_coco2yolo(seg, img_w, img_h)
            
            if yolo is None:
                continue

            anno_info = (image_name, category_id, yolo)
            anno_infos = anno_dict.get(image_id)
            if not anno_infos:
                anno_dict[image_id] = [anno_info]
            else:
                anno_infos.append(anno_info)
                anno_dict[image_id] = anno_infos

        return anno_dict

    def save_classes(self):
        sorted_classes = list(map(lambda x: x['name'], sorted(self.labels['categories'], key=lambda x: x['id'])))
        print('coco names', sorted_classes)
        with open('coco.names', 'w', encoding='utf-8') as f:
            for cls in sorted_classes:
                f.write(cls + '\n')
        f.close()

    def coco2yolo(self):
        print("loading image info...")
        images_info = self._load_images_info()
        print("loading done, total images", len(images_info))

        print("start converting...")
        anno_dict = self._convert_anno(images_info)
        print("converting done, total labels", len(anno_dict))

        print("saving txt file...")
        self._save_txt(anno_dict)
        print("saving done")
        
    def coco2yolo_seg(self):
        print("loading image info...")
        images_info = self._load_images_info()
        print("loading done, total images", len(images_info))

        print("start converting...")
        anno_dict = self._convert_anno_seg(images_info)
        print("converting done, total labels", len(anno_dict))

        print("saving txt file...")
        self._save_txt_seg(anno_dict)
        print("saving done")

    def _save_txt(self, anno_dict):
        for k, v in anno_dict.items():
            file_name = v[0][0].split(".")[0] + ".txt"
            with open(os.path.join(output, file_name), 'w', encoding='utf-8') as f:
                for obj in v:
                    cat_name = self.coco_id_name_map.get(obj[1])
                    category_id = self.coco_name_list.index(cat_name)
                    bbox = obj[2]
                    box = [f'{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}']
                    box = ' '.join(box)
                    line = str(category_id) + ' ' + box
                    f.write(line + '\n')

    def _save_txt_seg(self, anno_dict):
        for k, v in anno_dict.items():
            file_name = v[0][0].split(".")[0] + ".txt"
            with open(os.path.join(output, file_name), 'w', encoding='utf-8') as f:
                for obj in v:
                    cat_name = self.coco_id_name_map.get(obj[1])
                    category_id = self.coco_name_list.index(cat_name)
                    box = ['{:.6f} {:.6f}'.format(x[0], x[1]) for x in obj[2]]
                    box = ' '.join(box)
                    line = str(category_id) + ' ' + box
                    f.write(line + '\n')

if __name__ == '__main__':
    json_file = './dataset/coco/annotations/instances_default.json'
    
    seg = input("Do you want to convert bbox or segmentation? (b/s)")
    
    if seg == 's':
        output = './dataset/coco/labels_seg/'
        c2y = COCO2YOLO(json_file, output)
        c2y.coco2yolo_seg()
    elif seg == 'b':
        output = './dataset/coco/labels_box/'
        c2y = COCO2YOLO(json_file, output)
        c2y.coco2yolo()
    else:
        print("Please enter b or s")
