#!/usr/bin/env bash

docker run --volume $(pwd):/birdnet cmi/birdnet \
       python analyze_batch.py \
       --source-files "test/249BC3055C030FB4_20200308_181500_0.flac" \
       --dataset-path "/birdnet/test_outputs/birdnet" \
       --model-tag-path "/birdnet/model/BirdNET_Soundscape_Model.pkl" \
       --parameters '{"lat": 36, "lon": -121, "week": 24, "overlap": 0, "spp": 1, "sensitivity": 1.0, "min_conf": 0.000001}' \
       --source-dir "/birdnet/" \
       --filename-formats "site.yyyymmddHHMMSS"
