import cv2
import numpy as np
import random
import os
from pathlib import Path
from glob import glob
from tqdm import tqdm
from exif import Image
from sklearn.cluster import KMeans
from datetime import datetime
import shutil

random.seed(42)


def ask_user():
    
    delete = input("Do you want to delete the existing data? (y/n): ")
    if delete.lower() in ['y', 'n']:
        if delete.lower() == 'y':
            delete = True
        elif delete.lower() == 'n':
            delete = False
        else:
            print("Invalid input. Please enter 'y' or 'n'.")
            exit()
    
    # ask user for crop variable
    segment = input("Your labels are bbox or segmentation? (b/s): ")
    if segment.lower() in ['b', 's']:
        if segment.lower() == 's':
            segment = True
        elif segment.lower() == 'b':
            segment = False
        else:
            print("Invalid input. Please enter 'b' or 's'.")
            exit()
    else:
        print("Invalid input. Please enter 'b' or 's'.")
        exit()
        
    crop = input("Do you want to crop the images? (y/n): ")
    if crop.lower() in ['y', 'n']:
        if crop.lower() == 'y':
            crop = True
        else:
            crop = False
    else:
        print("Invalid input. Please enter 'y' or 'n'.")
        exit()
    
    splitting = input("How you want to split the images? (random, gps, date): ")
    if splitting.lower() in ['random', 'gps', 'date']:
        splitting = splitting.lower()
    else:
        print("Invalid input. Please enter 'random', 'gps' or 'date'.")
        exit()
        
    merge = input("Do you want to merge billboards classes into one? (y/n): ")
    if merge.lower() in ['y', 'n']:
        if merge.lower() == 'y':
            merge = True
        elif merge.lower() == 'n':
            merge = False
        else:
            print("Invalid input. Please enter 'y' or 'n'.")
            exit()
            
    exclude = input("Do you want to exclude the road class? (y/n): ")
    if exclude.lower() in ['y', 'n']:
        if exclude.lower() == 'y':
            exclude = True
        elif exclude.lower() == 'n':
            exclude = False
        else:
            print("Invalid input. Please enter 'y' or 'n'.")
            exit()
            
    return delete, segment, crop, splitting, merge, exclude

class Process:
    def __init__(self, inp_dir, segment, crop, splitting, merge, exclude):
        if segment:
            self.labels = sorted(list(Path(inp_dir+'/labels_seg/').glob('*.txt')))
        else:
            self.labels = sorted(list(Path(inp_dir+'/labels_box/').glob('*.txt')))
        self.convert_gps = lambda x: x[0] + x[1]/60 + x[2]/3600 
        
        print("total images with labels: ", len(self.labels))
        self.crop = crop
        self.splitting = splitting
        self.segment = segment
        self.merge = merge
        self.exclude = exclude
        
        if splitting == 'gps':
            feature = self.read_gps_data()
            self.clusters = self.infer_n_groups(feature)
            
        elif splitting == 'date':
            feature = self.read_date_data()
            self.clusters = self.infer_n_groups(feature)
        else:
            self.clusters = np.random.randint(0, 5, len(self.labels))
  
        print(f'unique folds: {np.unique(self.clusters)}')
            
    def infer_n_groups(self, feat):
        kmeans = KMeans(n_clusters=5, random_state=0).fit(feat)
        compontents = kmeans.labels_
        
        return compontents

    def read_gps_data(self):
        locations = []
        
        for p in self.labels:
            img_path = str(p).replace('.txt', '.JPG').replace('labels_seg', 'images').replace('labels_box', 'images')
            with open(img_path, 'rb') as src:
                img = Image(src)
                
                if type(img.gps_latitude) == float:
                    location = (img.gps_longitude, img.gps_latitude)
                else:
                    location = self.convert_gps(img.gps_latitude), self.convert_gps(img.gps_longitude)
                
                locations.append(location)        
        
        locations = np.array(locations, dtype=np.float32)
        return locations
        
    def read_date_data(self):
        dates = []
        
        for p in self.labels:
            img_path = str(p).replace('.txt', '.JPG').replace('labels_seg', 'images').replace('labels_box', 'images')
            with open(img_path, 'rb') as src:
                img = Image(src)
                
                date = datetime.strptime(img.datetime, '%Y:%m:%d %H:%M:%S')
                date = date.timestamp()
                
                
                dates.append(date)
                
        dates = np.array(dates).reshape(-1, 1)
        return dates
    
    def doit(self, output):
        base = Path(output)
        base.mkdir(exist_ok=True)
        
        kfolds_dict = {}

        for i in range(5):
            Path(output + f'/fold_{i}/').mkdir(exist_ok=True, parents=True)
            Path(output + '/data/' + f'/fold_{i}/').mkdir(exist_ok=True, parents=True)
            kfolds_dict[i] = []
        
        for i, p in tqdm(enumerate(self.labels)):
            img_p = str(p).replace('.txt', '.JPG').replace('labels_seg', 'images').replace('labels_box', 'images')

            img = cv2.imread(img_p)

            sub = f'fold_{self.clusters[i]}/'
            
            h, w = img.shape[:2]
            
            offset = max(0, int(h-w//2))
            
            if self.crop:
                if self.segment:
                    paths = self.perform_cut_segment(output, img, p, sub, offset)
                else:
                    paths = self.perform_cut(output, img, p, sub, offset)
            else:
                paths = self.perform(output, img, p, sub)
                
            kfolds_dict[self.clusters[i]] += paths
            
        # create train.txt, val.txt, test.txt per fold
        for i in range(5):
            test = i
            val = i+1 if i<4 else 0
            train = [i for i in range(5) if i not in [test, val]]
            
            with open(output + f'/fold_{i}/train.txt', 'w') as f:
                for t in train:
                    f.write('\n'.join(kfolds_dict[t]))
                
            with open(output + f'/fold_{i}/val.txt', 'w') as f:
                f.write('\n'.join(kfolds_dict[val]))
                
            with open(output + f'/fold_{i}/test.txt', 'w') as f:
                f.write('\n'.join(kfolds_dict[test]))
                
        with open(output + '/train.txt', 'w') as f:
            for i in range(5):
                f.write('\n'.join(kfolds_dict[i][10:-10]))
                f.write('\n')
                
        with open(output + '/val.txt', 'w') as f:
            for i in range(5):
                f.write('\n'.join(kfolds_dict[i][:10]))
                f.write('\n')
            
        with open(output + '/test.txt', 'w') as f:
            for i in range(5):
                f.write('\n'.join(kfolds_dict[i][-10:]))
                f.write('\n')
                
    def perform_cut(self, output, img, p, sub, offset):
        org_h, org_w = img.shape[:2]
        
        img = img[offset:, :, :]
        
        offseted_h, offseted_w = img.shape[:2]

        img_0 = img[:, :org_w//2, :]
        img_1 = img[:, org_w//2:, :]
        
        cv2.imwrite(output + '/data/' + sub + str(p.name).replace('.txt', '_0.jpg'), img_0)
        cv2.imwrite(output + '/data/' + sub + str(p.name).replace('.txt', '_1.jpg'), img_1)
        
        with open(str(p), 'r') as f:
            lines = f.readlines()

        a_left = []
        a_right = []

        for line in lines:
            line = line.split()
            cl = line[0]
            
            if self.merge and int(cl) == 1:
                cl = str(0)
                
            if self.exclude and int(cl) == 2:
                continue
            
            x, y, w, h = line[1:]
            
            x = float(x)*org_w
            y = float(y)*org_h
            w = float(w)*org_w
            h = float(h)*org_h

            y -= offset
            # filter box if it is outside of the image in 50% of its height
            if y < 0:
                if y+h/2 < 0:
                    continue
                else:
                    temp = y + h
                    y = temp/2
                    h = temp
                    
            if x < org_w//2:
                if x+w/2 > org_w//2:
                    temp = x+w-org_w/2-1
                    x = org_w/2 - 1 - temp/2 
                    w = temp
                    
                x = x / (org_w//2)
                y = y / offseted_h
                w = w / (org_w//2)
                h = h / offseted_h
                
                if x > 1.0:
                    print(x, y, w, h)
                    
                if x < 0.0:
                    print(x, y, w, h)
                    
                a_left.append([cl, x, y, w, h])
            
            if x > org_w//2:
                x = x-org_w//2
                
                if x+w/2 < 0:
                    temp = x + w
                    x = temp/2 
                    w = temp
                    
                x = x / (org_w//2)
                y = y / offseted_h
                w = w / (org_w//2)
                h = h / offseted_h
                
                if x > 1.0:
                    print(x, y, w, h)
                    
                if x < 0.0:
                    print(x, y, w, h)
                    
                a_right.append([cl, x, y, w, h])
            
        with open(output + '/data/' + sub + str(p.name).replace('.txt', '_0.txt'), 'w') as f:
            for z in a_left:
                z[1:] = [str(x) for x in z[1:]]
                f.write(' '.join(z)+'\n')

        with open(output + '/data/' + sub + str(p.name).replace('.txt', '_1.txt'), 'w') as f:
            for z in a_right:
                z[1:] = [str(x) for x in z[1:]]
                f.write(' '.join(z)+'\n')
                
        return [os.path.abspath(output + '/data/' + sub + str(p.name).replace('.txt', '_0.jpg')), os.path.abspath(output + '/data/' + sub + str(p.name).replace('.txt', '_1.jpg'))]
    
    @staticmethod
    def PolyArea(x,y):
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
    
    def perform_cut_segment(self, output, img, p, sub, offset):
        org_h, org_w = img.shape[:2]
        
        img = img[offset:, :, :]
        
        offseted_h, offseted_w = img.shape[:2]

        img_0 = img[:, :org_w//2, :]
        img_1 = img[:, org_w//2:, :]
        
        cv2.imwrite(output + '/data/' + sub + str(p.name).replace('.txt', '_0.jpg'), img_0)
        cv2.imwrite(output + '/data/' + sub + str(p.name).replace('.txt', '_1.jpg'), img_1)
        
        with open(str(p), 'r') as f:
            lines = f.readlines()

        a_left = []
        a_right = []

        for line in lines:
            line = line.split()
            cl = line[0]
            
            if self.merge and int(cl) == 1:
                cl = str(0)
                
            if self.exclude and int(cl) == 2:
                continue
            
            line = np.array([float(i)*org_w if k%2==0 else float(i)*org_h for k, i in enumerate(line[1:]) ]).reshape(-1, 2)

            area = self.PolyArea(line[:, 0], line[:, 1])

            line[:, 1] -= offset

            line_cp = line.copy()
            line_cp[line_cp[:, 1] < 0, 1] = 0

            area_cp = self.PolyArea(line_cp[:, 0], line_cp[:, 1])

            if area_cp/area < 0.5:
                continue

            line_cp = line.copy()
            line_cp[line_cp[:, 0] > org_w//2, 0] = org_w//2-1

            area_cp = self.PolyArea(line_cp[:, 0], line_cp[:, 1])
            
            if area_cp > 0.4 * area:
                line_cp = line.copy()
                line_cp[line_cp[:, 0]>=org_w//2, 0] = org_w//2-1
                line_cp /= np.array([offseted_w//2, offseted_w//2])

                line_cp[line_cp<0] = 0.0
                    
                a_left.append([cl] + list(line_cp.flatten()))
            
            line_cp = line.copy()
            line_cp[line_cp[:, 0] < org_w//2, 0] = 0

            area_cp = self.PolyArea(line_cp[:, 0], line_cp[:, 1])

            if area_cp > 0.3 * area:
                line_cp = line.copy()
                line_cp[:, 0] -= org_w//2
                line_cp[line_cp[:, 0]>=org_w//2, 0] = org_w//2-1
                line_cp /= np.array([offseted_w//2, offseted_w//2])

                line_cp[line_cp<0] = 0.0
                    
                a_right.append([cl] + list(line_cp.flatten()))
            
        with open(output + '/data/' + sub + str(p.name).replace('.txt', '_0.txt'), 'w') as f:
            for z in a_left:
                z[1:] = [str(x) for x in z[1:]]
                f.write(' '.join(z)+'\n')

        with open(output + '/data/' + sub + str(p.name).replace('.txt', '_1.txt'), 'w') as f:
            for z in a_right:
                z[1:] = [str(x) for x in z[1:]]
                f.write(' '.join(z)+'\n')
                
        return [os.path.abspath(output + '/data/' + sub + str(p.name).replace('.txt', '_0.jpg')), os.path.abspath(output + '/data/' + sub + str(p.name).replace('.txt', '_1.jpg'))]
                    
    def perform(self, output, img, p, sub):
        org_h, org_w = img.shape[:2]
        
        cv2.imwrite(output + '/data/' + sub + str(p.name).replace('.txt', '.jpg'), img)
        
        shutil.copy(str(p), output + '/data/' + sub + str(p.name))

        return [os.path.abspath(output + '/data/' + sub + str(p.name).replace('.txt', '.jpg'))]        

def prepare():
    inp_dir = './dataset/coco/'
    
    delete, segment, crop, splitting, merge, exclude = ask_user()
    
    if segment:
        out_dir = './dataset/yolo_seg/'
    else:
        out_dir = './dataset/yolo_box/'
        
    if delete:
        dirpath = Path(out_dir)
        if dirpath.exists() and dirpath.is_dir():
            shutil.rmtree(dirpath)
    
    Process(inp_dir, segment, crop, splitting, merge, exclude).doit(out_dir)
    
    
if __name__ == '__main__':
    prepare()
