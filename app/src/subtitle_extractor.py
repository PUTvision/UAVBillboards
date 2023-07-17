import subprocess as sp
import json
import re


class SubtitleRow:
    def __init__(self, gps: str, alt:float, h_speed: float, v_speed: float) -> None:
        self.h_speed = h_speed
        self.v_speed = v_speed
        self.alt = alt
        self.lon, self.lat = [float(x) for x in gps[5:-1].split(', ')][:2]

class SubtitleExtractor:
    def __init__(self) -> None:
        pass

    def __call__(self, video_path: str) -> dict:
        out = sp.run(['ffprobe','-of','json','-show_entries', 'format:stream', video_path],\
             stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True)

        results = json.loads(out.stdout)
        
        metadata_format = results['format']['tags']
        
        out = sp.run(['ffmpeg','-i',video_path, '-map', 's:0', '-f','webvtt','-'],\
             stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True)
        
        subtitle = out.stdout
        
        parsed_subtitle = self.__parse_subtitle(subtitle)
        
        return parsed_subtitle

    def __parse_subtitle(self, subtitle: str) -> dict:
        subtitle = subtitle.split('\n')[2:]
        
        parsed = {}
        
        for i in range(0, len(subtitle), 4):
            m, s = subtitle[i].split(' ')[0].split('.')[0].split(':')
            timestamp = int(m) * 60 + int(s)
            
            row = subtitle[i+1]
            
            gps = re.search(r'GPS \([0-9]+\.[0-9]+, [0-9]+\.[0-9]+, [0-9]+\)', row)[0]
            h_s = float(re.search(r'H.S -*[0-9]+\.[0-9]+', row)[0].split(' ')[1])
            v_s = float(re.search(r'V.S -*[0-9]+\.[0-9]+', row)[0].split(' ')[1])
            alt = float(re.search(r'H -*[0-9]+\.[0-9]+', row)[0].split(' ')[1])

            parsed[timestamp] = SubtitleRow(gps, alt, h_s, v_s)
            
        return parsed
            
            
        
