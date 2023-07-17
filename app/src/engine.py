import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pickle

from src.aggregator import Aggregator
from src.exif_extractor import ExifExtractor
from src.subtitle_extractor import SubtitleExtractor
from src.yolo import YOLO


class Engine:
    def __init__(self, config: dict,  video_path: str, yolo_model_path: str):
        self.config = config
        self.video_path = video_path

        self.video_info = {}
        self.geo_info = {}
        self.camera_info = {}
        self.detections = {}

        self._subtitle_extractor = SubtitleExtractor()
        self._extif_extractor = ExifExtractor()
        self._aggregator = Aggregator(config)
        
        self.model = YOLO(yolo_model_path)

        self.frame_counter = -1

    def parse_video_info(self):
        print('Parsing video info...')

        stream = cv2.VideoCapture(self.video_path)

        self.video_info = {
            'fps': float(stream.get(cv2.CAP_PROP_FPS)),
            'width': int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'length': int(stream.get(cv2.CAP_PROP_FRAME_COUNT) / stream.get(cv2.CAP_PROP_FPS)),
            'name': self.video_path.split('/')[-1],
        }

        stream.release()

    def parse_geo_info(self):
        print('Parsing geo info from subtitles...')
        self.geo_info = self._subtitle_extractor(self.video_path)

    def parse_camera_info(self):
        print('Parsing camera info from exif...')
        self.camera_info = self._extif_extractor(self.video_path)

    def process_frames(self, show: bool = False, skip: int = 0) -> dict:
        print('Processing frames...')
        stream = cv2.VideoCapture(self.video_path)

        detections = {}

        progress_bar = tqdm(
            total=self.video_info['length'] * self.video_info['fps'])

        while True:
            ret, frame = stream.read()

            if not ret:
                break

            self.frame_counter += 1
            progress_bar.update(1)
            
            if self.frame_counter < skip*self.video_info['fps']:
                continue

            results = self.model.process_frame(frame)

            if show and self.frame_counter % 10 == 0:
                frame = self.model.draw_results(frame, results)
                cv2.imshow('frame', cv2.resize(frame, None, fx=0.4, fy=0.4))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            for k, v in results.items():
                if k in detections:
                    detections[k].append((self.frame_counter, v))
                else:
                    detections[k] = [(self.frame_counter, v)]

        if show:
            cv2.destroyAllWindows()

        stream.release()
        progress_bar.close()

        self.detections = detections

    def aggregate_results(self):
        print('Aggregating results...')

        if not self.video_info or not self.camera_info or not self.geo_info or not self.detections:
            raise Exception(
                'Missing data. Run parse_video_info, parse_camera_info, parse_geo_info and process_frames first.')       

        self.results = self._aggregator(
            self.video_path, self.detections, self.video_info, self.camera_info, self.geo_info)

    def save_results(self, path: str):
        print('Generating files...')

        if self.results is None:
            raise Exception('Missing data. Run aggregate_results first.')

        Path('./outputs').mkdir(parents=True, exist_ok=True)
        report_path = './outputs/' + path.split('/')[-1] + '.pkl'

        pickle.dump(self.results, open(report_path, 'wb'))
