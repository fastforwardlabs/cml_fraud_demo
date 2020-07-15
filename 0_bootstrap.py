## Run this file to auto deploy the model.

### Install the requirements
!bash cdsw-build.sh

### Download the data file and save it in the specified directory


from cmlbootstrap import CMLBootstrap
from datetime import datetime
import os, time

from IPython.display import Javascript, HTML
import os
import time
import json
import requests
import xml.etree.ElementTree as ET

# Set the setup variables needed by CMLBootstrap
HOST = os.getenv("CDSW_API_URL").split(
    ":")[0] + "://" + os.getenv("CDSW_DOMAIN")
USERNAME = os.getenv("CDSW_PROJECT_URL").split(
    "/")[6]  # args.username  # "vdibia"
API_KEY = os.getenv("CDSW_API_KEY") 
PROJECT_NAME = os.getenv("CDSW_PROJECT")  

# Instantiate API Wrapper
cml = CMLBootstrap(HOST, USERNAME, API_KEY, PROJECT_NAME)

# Set the STORAGE environment variable
try : 
  storage=os.environ["STORAGE"]
except:
  tree = ET.parse('/etc/hadoop/conf/hive-site.xml')
  root = tree.getroot()
    
  for prop in root.findall('property'):
    if prop.find('name').text == "hive.metastore.warehouse.dir":
        storage = prop.find('value').text.split("/")[0] + "//" + prop.find('value').text.split("/")[2]
  storage_environment_params = {"STORAGE":storage}
  storage_environment = cml.create_environment_variable(storage_environment_params)
  os.environ["STORAGE"] = storage

# Upload the data to the cloud storage

!unzip -o data/creditcardfraud.zip

!hdfs dfs -mkdir -p $STORAGE/datalake
!hdfs dfs -mkdir -p $STORAGE/datalake/data
!hdfs dfs -mkdir -p $STORAGE/datalake/data/anomalydetection
!hdfs dfs -copyFromLocal /home/cdsw/data/creditcard.csv $STORAGE/datalake/data/anomalydetection/creditcard.csv

!rm /home/cdsw/data/creditcard.csv
