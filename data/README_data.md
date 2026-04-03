Data Downloading and Initial Processing Instructions

1)  NLCD Land Cover (CONUS) Data
        Can be downloaded directly from https://www.mrlc.gov/data
            -   most recently downloaded on 3/13/2026
        Downloaded data should be placed in data/raw/nlcd/
        To shrink file size, data clipped down to a bounding box surrounding the northeast US by clip_nlcd.py

    ** From root project directory, run command
        python scripts/clip_nlcd.py --all --input-dir "your/path/to/downloaded/nlcd/files"

2)  USGS National Elevation Dataset (NED)
        Downloaded programmatically to data/raw/ned/
            -   most recently downloaded on 4/3/2026
        USGS 1 Arc Second Product chosen, tiles covering the northeast US.
        For convenience of access, files are then merged into a larger mosaic by prepare_dem.py
    
    ** From root project directory, run commands
        python scripts/download_ned.py
        python scripts/prepare_dem.py

3)  USA National Phenology Network Data
        Programmatically found, preprocessed, and downloaded by download_npn.py
            -   most recently downloaded on 4/1/2026
        Individual observations saved in data/raw/observer_reports as npn_vermont.csv
        Based on specific phenology observed (colored leaves/falling leaves) each observation is labeled as pre-transition/early/peak/late

    ** From root project directory, run command
        python scripts/download_npn.py --request-source "Your Name" --states ME NH VT MA RI CT NY NJ PA
    
        After downloading, observations at the same date/time are consolidated into pooled observations. 
        These are saved in data/processed/observer_reports as observations.csv
        Further, ARD tiles which contain observations are determined.
        These are saved in data/processed/observer_reports as ard_tile_list.txt
    
    ** From root project directory, run command 
        python scripts/process_observations.py

4)  Landsat ARD Data
        Programmatically found, preprocessed, and downloaded by preprocess_landsat.py
            -   most recently downloaded on 4/2/2026
        Data located using STAC queries, then preprocessed and downloaded from the usgs-landsat AWS S3 bucket (requester pays). Setting up local AWS authentication in advance is required.
        By default, data limited to 
            tiles in the northeast containing observations, 
            August, September, October, November,
            less than 60% cloud cover.
        
    ** From root project directory, run command
        python scripts/preprocess_landsat.py --tile-list data/processed/observations/ard_tile_list.txt
    
        It is highly recommended to run this command in an AWS EC2 instance in us-west so as to be colocated with the usgs-landsat AWS S3 bucket.



