# UAV Billboards - application

The instructions below are for running the application.

## System Requirements

* `ffmpeg` - for video conversion

    ```bash
    sudo apt install ffmpeg
    ```

* `exiftool` - for metadata extraction

    ```bash
    sudo apt install libimage-exiftool-perl
    ```

## Python Requirements

Install the required python packages using `pip`:

```bash
pip install -r requirements.txt
```

## Running the Application

* Modify the `config.yml` file with your camera settings.

* Run the application. It should be run for each video file. It generates configs and metadata files for each video file in the `./outputs/` directory.

    ```bash
    Usage: main.py [OPTIONS]

    Options:
    --video TEXT    Path to video  [required]
    --show          Show video
    --skip INTEGER  Skip seconds
    --help          Show this message and exit.
    ```

    Example:

    ```bash
    python3 main.py --video ./data/example.MP4
    ```

* While you process the video files, you can run the `update_map.py` to generate the results in the html file.

    ```bash
    Usage: update_map.py [OPTIONS]

    Options:
    --report TEXT  Report path  [required]
    --help         Show this message and exit.
    ```

    Example:

    ```bash
    python3 update_map.py --report data/example_map.html
    ```

    > **Note:** The script uses Nominatim to reverse geocode the coordinates. If you want to use it, you need to set the `NOMINATIM_KEY` variable. You can get an API key from [here](https://developer.mapquest.com/).
