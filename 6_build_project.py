## Run this file to auto deploy the model, run a job, and deploy the application

### Install the requirements
!bash cdsw-build.sh
!pip3 install dash

# Create the directories and upload data

# build the project
from cmlbootstrap import CMLBootstrap
from IPython.display import Javascript, HTML
import os
import time
import json
import requests
import xml.etree.ElementTree as ET
import datetime

run_time_suffix = datetime.datetime.now()
run_time_suffix = run_time_suffix.strftime("%d%m%Y%H%M%S")


HOST = os.getenv("CDSW_API_URL").split(
    ":")[0] + "://" + os.getenv("CDSW_DOMAIN")
USERNAME = os.getenv("CDSW_PROJECT_URL").split(
    "/")[6]  # args.username  # "vdibia"
API_KEY = os.getenv("CDSW_API_KEY") 
PROJECT_NAME = os.getenv("CDSW_PROJECT")  

# Instantiate API Wrapper
cml = CMLBootstrap(HOST, USERNAME, API_KEY, PROJECT_NAME)

# set the storage variable to the default location
try : 
  s3_bucket=os.environ["STORAGE"]
except:
  tree = ET.parse('/etc/hadoop/conf/hive-site.xml')
  root = tree.getroot()
    
  for prop in root.findall('property'):
    if prop.find('name').text == "hive.metastore.warehouse.dir":
        s3_bucket = prop.find('value').text.split("/")[0] + "//" + prop.find('value').text.split("/")[2]
  storage_environment_params = {"STORAGE":s3_bucket}
  storage_environment = cml.create_environment_variable(storage_environment_params)
  os.environ["STORAGE"] = s3_bucket

!unzip data/creditcardfraud.zip -d data

!hdfs dfs -mkdir -p $STORAGE/datalake
!hdfs dfs -mkdir -p $STORAGE/datalake/data
!hdfs dfs -mkdir -p $STORAGE/datalake/data/anomalydetection
!hdfs dfs -copyFromLocal /home/cdsw/data/creditcard.csv $STORAGE/datalake/data/anomalydetection/creditcard.csv

!rm /home/cdsw/data/creditcard.csv

  
# This will run the data ingest file. You need this to create the hive table from the 
# csv file.
exec(open("1_data_ingest.py").read())

# Get User Details
user_details = cml.get_user({})
user_obj = {"id": user_details["id"], 
            "username": user_details["username"],
            "name": user_details["name"],
            "type": user_details["type"],
            "html_url": user_details["html_url"],
            "url": user_details["url"]
            }

# Get Project Details
project_details = cml.get_project({})
project_id = project_details["id"]

# Get Default Engine Details
default_engine_details = cml.get_default_engine({})
default_engine_image_id = default_engine_details["id"]

# Create Job
create_jobs_params = {"name": "Train Model " + run_time_suffix,
                      "type": "manual",
                      "script": "3_model_train.py",
                      "timezone": "America/Los_Angeles",
                      "environment": {},
                      "kernel": "python3",
                      "cpu": 1,
                      "memory": 2,
                      "nvidia_gpu": 0,
                      "include_logs": True,
                      "notifications": [
                          {"user_id": user_obj["id"],
                           "user":  user_obj,
                           "success": False, "failure": False, "timeout": False, "stopped": False
                           }
                      ],
                      "recipients": {},
                      "attachments": [],
                      "include_logs": True,
                      "report_attachments": [],
                      "success_recipients": [],
                      "failure_recipients": [],
                      "timeout_recipients": [],
                      "stopped_recipients": []
                      }

new_job = cml.create_job(create_jobs_params)
new_job_id = new_job["id"]
print("Created new job with jobid", new_job_id)

# Start a job
job_env_params = {}
start_job_params = {"environment": job_env_params}
job_id = new_job_id
job_status = cml.start_job(job_id, start_job_params)
print("Job started")

# Wait for the model training job to complete
model_traing_completed = False
while model_traing_completed == False:
  if cml.get_jobs({})[0]['latest']['status'] == 'succeeded':
    print("Model training Job complete")
    break
  else:
    print ("Model training Job running.....")
    time.sleep(10)

# Create Model
example_model_input = {"v": [-1.3598071336738,-0.0727811733098497,2.53634673796914,1.37815522427443,-0.338320769942518,0.462387777762292,0.239598554061257,0.0986979012610507,0.363786969611213,0.0907941719789316,-0.551599533260813,-0.617800855762348,-0.991389847235408,-0.311169353699879,1.46817697209427,-0.470400525259478,0.207971241929242,0.0257905801985591,0.403992960255733,0.251412098239705,-0.018306777944153,0.277837575558899,-0.110473910188767,0.0669280749146731,0.128539358273528,-0.189114843888824,0.133558376740387,-0.0210530534538215], 
                    "time": 0, "amount": 149.62}

create_model_params = {
    "projectId": project_id,
    "name": "Fraud Detection " + run_time_suffix,
    "description": "Fraud Detection",
    "visibility": "private",
    "targetFilePath": "4_model_deploy.py",
    "targetFunctionName": "predict",
    "engineImageId": default_engine_image_id,
    "kernel": "python3",
    "examples": [
        {
            "request": example_model_input,
            "response": {}
        }],
    "cpuMillicores": 1000,
    "memoryMb": 2048,
    "nvidiaGPUs": 0,
    "authEnabled": False,
    "replicationPolicy": {"type": "fixed", "numReplicas": 1},
    "environment": {}}

new_model_details = cml.create_model(create_model_params)
access_key = new_model_details["accessKey"]  # todo check for bad response
model_id = new_model_details["id"]

print("New model created with access key", access_key)

#Wait for the model to deploy.
is_deployed = False
while is_deployed == False:
  model = cml.get_model({"id": str(new_model_details["id"]), "latestModelDeployment": True, "latestModelBuild": True})
  if model["latestModelDeployment"]["status"] == 'deployed':
    print("Model is deployed")
    break
  else:
    print ("Deploying Model.....")
    time.sleep(10)
    
## Change the line in the flask/single_view.html file.
import subprocess
subprocess.call(["sed", "-i",  's/access_key\s=.*/access_key = "' + access_key + '";/', "/home/cdsw/5_application.py"])


# Create Application
create_application_params = {
    "name": "Fraud App",
    "subdomain": run_time_suffix[:],
    "description": "Fraud web application",
    "type": "manual",
    "script": "5_application.py", "environment": {},
    "kernel": "python3", "cpu": 1, "memory": 2,
    "nvidia_gpu": 0
}

new_application_details = cml.create_application(create_application_params)
application_url = new_application_details["url"]
application_id = new_application_details["id"]

# print("Application may need a few minutes to finish deploying. Open link below in about a minute ..")
print("Application created, deploying at ", application_url)

#Wait for the application to deploy.
is_deployed = False
while is_deployed == False:
#Wait for the application to deploy.
  app = cml.get_application(str(application_id),{})
  if app["status"] == 'running':
    print("Application is deployed")
    break
  else:
    print ("Deploying Application.....")
    time.sleep(10)

HTML("<a href='{}'>Open Application UI</a>".format(application_url))    
