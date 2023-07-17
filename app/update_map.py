import cv2
import click
import pickle
from tqdm import tqdm
from glob import glob

from src.report_generator import ReportGenerator

@click.command()
@click.option('--report', required=True, type=str, help='Report path')
def update_map(report):
    
    files = sorted(glob('./outputs/*.pkl'))
    print(f'Found {len(files)} reports.')
    
    rg = ReportGenerator()
    
    for file in tqdm(files):
        results = pickle.load(open(file, 'rb'))
        rg(results)
    
    rg.save(report)
    print('Done.')
    
if __name__ == '__main__':
    update_map()
