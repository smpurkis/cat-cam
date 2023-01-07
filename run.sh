#!/usr/bin/env bash -x
cd /home/pi/projects/cat_cam
source /home/pi/projects/cat_cam/venv/bin/activate
export PORT="5000"
export SUBDOMAIN="purkis-cat-cam"
gunicorn --threads 12 --workers 2 -b "0.0.0.0:${PORT}" app:server & 
lt --port ${PORT} --subdomain ${SUBDOMAIN} --timeout 1 --local-host "0.0.0.0" -o 
