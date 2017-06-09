#!/bin/bash

echo "Running... (in background)"
python model_naive.py > txt/model_naive.txt &
python model_basic.py > txt/model_basic.txt &
python model_delta.py > txt/model_delta.txt &
python model_concat.py > txt/model_concat.txt &
python model_timeless.py > txt/model_timeless.txt &
