import folium
import os
from folium import IFrame
import numpy as np
from geopy.geocoders import Nominatim
import cv2
import base64


class ReportGenerator:
    def __init__(self):
        
        self.m = None
        self.geolocator = None
            
        if os.environ.get('NOMINATIM_KEY') is not None:
            self.geolocator = Nominatim(user_agent=os.environ.get('NOMINATIM_KEY'))

    def __call__(self, data: list[dict]):      

        for row in data:
            if self.m is None:
                self.m = folium.Map(location=[row['b_lat'], row['b_lon']], zoom_start=13)
            
            img = row['b_img']
            
            if self.geolocator is not None:
                location = str(self.geolocator.reverse(f"{row['b_lat']}, {row['b_lon']}").address)
            else:
                location = 'Location not available. Set NOMINATIM_KEY environment variable to enable reverse geocoding.'
            
            popup = self.popup_html(
                row['b_id'],
                row['b_date'],
                'free-standing' if row['b_type'] == 0 else 'wall-mounted',
                str(row['b_lat']),
                str(row['b_lon']),
                location,
                self.img_to_html(img),
                str(int(400/img.shape[1]*img.shape[0])),
            )
            
            pushpin = folium.features.CustomIcon('./data/billboard.png', icon_size=(30, 30))            
            folium.Marker([row['b_lat'], row['b_lon']], icon=pushpin, popup=popup).add_to(self.m)

    def save(self, path: str):
        self.m.save(path)

    @staticmethod
    def img_to_html(image: np.ndarray) -> str:

        image[np.all(image==(0,0,0), axis=2)] = (255,255,255)
        encoded = base64.b64encode(cv2.imencode('.jpg', image)[1].tobytes()).decode('UTF-8')

        return encoded

    @staticmethod
    def popup_html(b_id,
                   b_date,
                   b_type,
                   b_lat,
                   b_lon,
                   b_place,
                   b_img,
                   b_h,
                   ):

        html = """<!DOCTYPE html>
    <html>
    <head>
    <style type="text/css">
    .tg  {border-collapse:collapse;border-spacing:0;}
    .tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
    overflow:hidden;padding:10px 5px;word-break:normal;}
    .tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
    font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
    .tg .tg-baqh{text-align:center;vertical-align:top}
    .tg .tg-b3sw{background-color:#efefef;font-weight:bold;text-align:left;vertical-align:top}
    .tg .tg-0lax{text-align:left;vertical-align:top}
    </style>
    <table class="tg">
    <thead>
    <tr>
        <th class="tg-b3sw">Identificator</th>
        <th class="tg-0lax">""" + b_id + """</th>
    </tr>
    </thead>
    <tbody>
        <tr>
        <td class="tg-b3sw">Creation date</td>
        <td class="tg-0lax">""" + b_date + """</td>
    </tr>
    <tr>
        <td class="tg-b3sw">Type</td>
        <td class="tg-0lax">""" + b_type + """</td>
    </tr>
    <tr>
        <td class="tg-b3sw">Localization</td>
        <td class="tg-0lax">""" + b_lat + """, """ + b_lon + """</td>
    </tr>
    <tr>
        <td class="tg-b3sw">Place</td>
        <td class="tg-0lax">""" + b_place + """<br><a href="https://www.google.com/maps/place/32%C2%B012'47.6%22N+53%C2%B019'23.6%22E" target="_blank" rel="noopener noreferrer">Google Maps link</a></td>
    </tr>
    <tr>
        <td class="tg-b3sw">Image</td>
        <td class="tg-0lax"><br><img src="data:image/png;base64,""" + b_img + """" width="400" height=\"""" + b_h + """"></td>
    </tr>
    </tbody>
    </table>
    </html>
    """
        return html
