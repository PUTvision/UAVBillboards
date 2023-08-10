# UAV Billboards - ML training

The instructions below are for training the model and evaluating it on the test set. 

## Requirements

* `python3` - for running the scripts

    ```bash
    sudo apt install python3
    ```

* `ffmpeg` - for video conversion

    ```bash
    sudo apt install ffmpeg
    ```

* `exiftool` - for metadata extraction

    ```bash
    sudo apt install libimage-exiftool-perl
    ```

## Python dependencies

Install the required python packages using `pip`:

```bash
pip install -r requirements.txt
```

## Dataset

* Download dataset using `download.py` script:

    ```bash
    python3 dataset/download.py
    ```

* Convert dataset from COCO to YOLO format:

    > **Note:** You have to select do you want to generate bounding boxes (`b`) or segmentation masks (`s`). We used segmentation masks in our experiments.

     ```bash
    python3 dataset/convert.py
    ```

* Split dataset into train and test sets:

    > **Note:** The script below provides many options. Note that we use them in the following way: `y` (delete the existing data), `s` (segmentation masks), `y` (crop the images), `gps` (split the images based on geotags), `n` (no merge billboard classes), `n` (no exclude road-sign class).

    ```bash
    python3 dataset/prepare.py
    ```

## Training

YOLOv7 and YOLOv8 models are training using the scripts and configs from the `yolov7` and `yolov8` directories.

