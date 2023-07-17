from typing import Any
import cv2
import numpy as np
from datetime import datetime
import pyproj
from geographiclib.geodesic import Geodesic


class Aggregator:
    def __init__(self, config) -> None:

        self.geod = Geodesic.WGS84
        self.geodesic = pyproj.Geod(ellps='WGS84')

        self.field_of_view = config['camera']['fov']
        self.camera_matrix = np.array(
            config['camera']['camera_matrix']).reshape(3, 3)
        self.dist_matrix = np.array(
            config['camera']['dist_matrix'])
        self.new_camera_matrix = np.array(
            config['camera']['new_camera_matrix']).reshape(3, 3)

    def __call__(self, video_path, detections, video_info, camera_info, geo_info) -> Any:

        base_name = video_path.split('/')[-1].split('.')[0]

        w_image, h_image = video_info['width'], video_info['height']  # pixels
        HFOV, VFOV = np.deg2rad(self.field_of_view[0]), np.deg2rad(
            self.field_of_view[1])
        gimbal_pitch = np.deg2rad(90 + camera_info['camera_pitch'])
        gimbal_yaw = np.deg2rad(camera_info['camera_yaw'])

        locs = np.empty((len(geo_info.keys()), 2))
        altss = np.empty((len(geo_info.keys())))

        for i, row in enumerate(geo_info.values()):
            locs[i][0] = row.lat
            locs[i][1] = row.lon
            altss[i] = row.alt

        poly_loc_fn = np.poly1d(np.polyfit(locs[:, 0], locs[:, 1], 2))

        lats = np.linspace(locs[0][0], locs[-1][0],
                           int(len(locs)*video_info['fps']))

        alts = np.concatenate(np.array([
            np.linspace(altss[k], altss[k+1], int(video_info['fps'])) for k in range(len(altss)-1)
        ]), axis=0)

        results = []

        for i, key in enumerate(detections.keys()):
            hypotesys = []

            middle_det = detections[key][len(detections[key])//2]

            classs = middle_det[1].class_id
            roi = middle_det[1].roi_masked

            for det_id in range(1, len(detections[key])):
                act_det = detections[key][det_id]

                prev_lat, prev_lon = lats[act_det[0] -
                                          1], poly_loc_fn(lats[act_det[0]-1])
                lat, lon = lats[act_det[0]], poly_loc_fn(lats[act_det[0]])

                if act_det[0] > len(alts)-1:
                    continue

                alt = alts[act_det[0]]

                billboard_image_center = np.array(
                    [
                        act_det[1].image_center[0],
                        act_det[1].image_center[1] + act_det[1].size[1]//2
                    ])

                billboard_image_center = cv2.undistortPoints(billboard_image_center.reshape(1, 1, 2).astype(
                    np.float32), self.camera_matrix, self.dist_matrix, None, self.new_camera_matrix)
                billboard_image_center = billboard_image_center[0][0]

                billboard_w_from_center = billboard_image_center[0]-(w_image/2)
                billboard_h_from_bottom = h_image-billboard_image_center[1]

                alpha = (billboard_h_from_bottom / h_image) * \
                    (gimbal_pitch + VFOV/2)
                ground_dy = alt * np.tan(alpha)

                beta = (billboard_w_from_center/(w_image/2)) * \
                    (HFOV/2 - gimbal_yaw)

                d = ground_dy/np.cos(beta) + alt*np.tan(gimbal_pitch - VFOV/2)

                fwd_azimuth, _, _ = self.geodesic.inv(
                    prev_lon, prev_lat, lon, lat)

                g = self.geod.Direct(
                    lat, lon, fwd_azimuth + np.rad2deg(beta), d)

                hypotesys.append([g['lat2'], g['lon2']])

            pred_lat, pred_lon = np.mean(hypotesys, axis=0)

            results.append({
                'b_id': f'{base_name}__{key}',
                'b_date': datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                'b_type': classs,
                'b_lat': pred_lat,
                'b_lon': pred_lon,
                'b_img': roi,
            })

        return results
