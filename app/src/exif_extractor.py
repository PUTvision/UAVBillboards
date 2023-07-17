import json
import subprocess

class ExifExtractor:
    def __init__(self):
        pass
    
    def __call__(self, video_path: str) -> dict:
        process = subprocess.Popen(["exiftool", '-json', video_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()
        # lines = out.decode("utf-8")
        lines = json.loads(out)[0]

        return {
            'camera_pitch': float(lines['CameraPitch']),
            'camera_roll': float(lines['CameraRoll']),
            'camera_yaw': float(lines['CameraYaw']),
        }
