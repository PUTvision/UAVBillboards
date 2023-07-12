import requests
from zipfile import ZipFile


def download():
    "Download the dataset zip file from server and extract it to this folder."
    
    data_url = "https://chmura.put.poznan.pl/s/lIMsy8OlOjuXAIJ/download"
    
    print("Downloading dataset...")
    r = requests.get(data_url, stream=True)
    with open("./dataset/UAVBillboardsDataset.zip", "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                
    print("Extracting dataset...")
    with ZipFile("./dataset/UAVBillboardsDataset.zip", "r") as zipObj:
        zipObj.extractall('./dataset/coco')
    
    print("Done!")


if __name__ == "__main__":
    download()
