Data Downloading and Initial Processing Instructions

1)  NLCD Land Cover (CONUS) Data
        Can be downloaded directly from https://www.mrlc.gov/data
            -   most recently done so on 3/13/2026
        To shrink file size, data clipped down to a bounding box surrounding Vermont by clip_nlcd.py

    ** From root project directory, run command
        python scripts/clip_nlcd.py --all --input-dir "your/path/to/downloaded/nlcd/files"

2)  Landsat ARD Data
        Programmatically found, preprocessed, and downloaded by preprocess_landsat.py
            -   most recent download on 3/15/2026
        Data located using STAC queries, then preprocessed and downloaded from the usgs-landsat AWS S3 bucket (requester pays). Setting up local AWS authentication in advance is required.
        By default, data limited to 
            tiles intersecting Vermont, 
            August, September, October, November,
            less than 60% cloud cover.
        
    ** From root project directory, run command
        python scripts/preprocess_landsat.py

3)  USA National Phenology Network Data
        Programmatically found, preprocessed, and downloaded by download_npn.py
            -   most recent download on 3/27/2026
        Individual observations saved in /raw/observer_reports as npn_vermont.csv
        Based on specific phenology observed (colored leaves/falling leaves) each observation is labeled as pre-transition/early/peak/late

    ** From root project directory, run command
        python scripts/download_npn.py --request-source "INPUT_YOUR_NAME_HERE"
