from ultralytics import YOLO as YOLOv8
import numpy as np
from dataclasses import dataclass
import cv2

@dataclass
class YOLOResult:
    image_center: tuple[int, int]
    size: tuple[int, int]
    points: np.ndarray
    class_id: int
    confidence: float
    roi: np.ndarray
    roi_masked: np.ndarray

    @property
    def area(self) -> float:
        return float(self.size[0]) * float(self.size[1])
    
    @property
    def area_masked(self) -> float:
        return 0.5*np.abs(np.dot(self.points[:, 0],np.roll(self.points[:, 1],1))-np.dot(self.points[:, 1],np.roll(self.points[:, 0],1)))
    
    @property
    def diagonal(self) -> float:
        return np.sqrt(float(self.size[0]) ** 2 + float(self.size[1]) ** 2)
    
    @property
    def diagonal_masked(self) -> float:
        return np.sqrt(np.sum(np.square(self.points[0] - self.points[2])))

class YOLO:
    def __init__(self, yolo_model_path: str) -> None:
        self.model = YOLOv8(yolo_model_path)

    def process_frame(self, frame: np.ndarray) -> None:
        offset = max(0, int(frame.shape[0]-frame.shape[1]//2))
        
        frame = frame[240:]
           
        h, w = frame.shape[:2]
           
        results = self.model.track(frame, conf=0.75, device=0, half=True, show=False, verbose=False, tracker="bytetrack.yaml", persist=True)[0]

        predictions = {}
        
        for result in results:
            box = result.boxes
            
            id_cls = int(box.cls.cpu()[0].cpu().item())
            
            if id_cls not in [0, 1]:
                continue
            
            if box.id is None:
                continue
            
            # box
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            y1=y1 + offset
            y2=y2 + offset
            
            conf = box.conf.cpu()[0].cpu().item()
            roi = frame[y1-offset:y2-offset, x1:x2].copy()
            
            if (x2-x1) < 50 or (y2-y1) < 50:
                continue
            
            # mask
            mask = result.masks
                        
            points = mask.xy[0].astype(int)
            points[:, 1] += offset
            
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            
            roi_masked = frame[y_min-offset:y_max-offset, x_min:x_max].copy()
            local_mask = np.ones((y_max-y_min, x_max-x_min), dtype=np.uint8)
            cv2.fillPoly(local_mask, [np.array([(int(x-x_min), int(y-y_min)) for x, y in points], dtype=np.int64)], 0)
            local_mask = local_mask.astype(bool)
            roi_masked[local_mask] = 0
                    
            predictions[f'{int(box.id.cpu()[0].cpu().item())}'] = \
                YOLOResult(
                    image_center=(int((x1+x2)/2), int((y1+y2)/2)),
                    size=(int(x2-x1), int(y2-y1)),
                    points=points,
                    class_id=id_cls,
                    confidence=conf,
                    roi=roi,
                    roi_masked=roi_masked,
                )
                
        return predictions

    def draw_results(self, frame, results):
        
        for id, yolo_result in results.items():
            x, y = yolo_result.image_center
            w, h = yolo_result.size
            
            cv2.rectangle(frame, (x-w//2, y-h//2), (x+w//2, y+h//2), (0, 0, 255), 4)
            cv2.rectangle(frame, (x-w//2, y-h//2-10), (x+w//2, y-h//2), (0, 0, 255), -1)
            cv2.putText(frame, f'{id}', (x-w//2, y-h//2-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
            cv2.polylines(frame, [yolo_result.points], True, (0, 0, 255), 4)
        
        return frame
