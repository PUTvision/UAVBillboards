import click
import yaml
from src.engine import Engine

with open('./config.yml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


@click.command()
@click.option('--video', required=True, type=str, help='Path to video')
@click.option('--show', default=False, type=bool, is_flag=True, help='Show video')
@click.option('--skip', default=0, type=int, help='Skip seconds')
def main(video: str, show: bool, skip: int):
    
    engine = Engine(config, video_path=video, yolo_model_path='./data/best.pt')
    
    engine.parse_video_info()
    engine.parse_camera_info()
    engine.parse_geo_info()
    engine.process_frames(show, skip)
    
    engine.aggregate_results()
    engine.save_results(video)
    
    
if __name__ == '__main__':
    main()

