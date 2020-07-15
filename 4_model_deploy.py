## Part 4: Model Serving
#
# This script explains how to create and deploy Models in CML which function as a 
# REST API to serve predictions. This feature makes it very easy for a data scientist 
# to make trained models available and usable to other developers and data scientists 
# in your organization.
#
### Requirements
# Models have the same requirements as Experiments:
# - model code in a `.py` script, not a notebook
# - a `requirements.txt` file listing package dependencies
# - a `cdsw-build.sh` script containing code to install all dependencies
#
# > In addition, Models *must* be designed with one main function that takes a dictionary as its sole argument
# > and returns a single dictionary.
# > CML handles the JSON serialization and deserialization.

# In this file, there is minimal code since calculating predictions is much simpler 
# than training a machine learning model.
# This script loads and uses the `cc_scaler.pkl` file for the MinMaxScaler and the `creditcard-fraud.model` file for the 
# pytorch model. 
# When a Model API is called, CML will translate the input and returned JSON blobs to and from python dictionaries.
# Thus, the script simply loads the model we saved at the end of the last notebook,
# passes the input dictionary into the model, and returns the results as a dictionary with the following format:
#    
#    {
#       "result" : loss.item()>split_point
#    }
#
# The Model API will return this dictionary serialized as JSON.
# 
### Creating and deploying a Model
# To create a Model using our `4_model_deploy.py` script, use the following settings:
# * **Name**: Fraud Detection
# * **Description**: Deep Anomaly Detection for Fraud
# * **File**: 4_model_deploy.py
# * **Function**: predict
# * **Input**: 
# ```
# {
#   "v": [
#     -1.3598071336738,
#     -0.0727811733098497,
#     2.53634673796914,
#     1.37815522427443,
#     -0.338320769942518,
#     0.462387777762292,
#     0.239598554061257,
#     0.0986979012610507,
#     0.363786969611213,
#     0.0907941719789316,
#     -0.551599533260813,
#     -0.617800855762348,
#     -0.991389847235408,
#     -0.311169353699879,
#     1.46817697209427,
#     -0.470400525259478,
#     0.207971241929242,
#     0.0257905801985591,
#     0.403992960255733,
#     0.251412098239705,
#     -0.018306777944153,
#     0.277837575558899,
#     -0.110473910188767,
#     0.0669280749146731,
#     0.128539358273528,
#     -0.189114843888824,
#     0.133558376740387,
#     -0.0210530534538215
#   ],
#   "time": 0,
#   "amount": 149.62
# }
# ```
# * **Kernel**: Python 3
# * **Engine Profile**: 1vCPU / 2 GiB Memory (**Note:** no GPU needed for scoring)
#
# The rest can be left as is.
#
# After accepting the dialog, CML will *build* a new Docker image using `cdsw-build.sh`,
# then *assign an endpoint* for sending requests to the new Model.

## Testing the Model
# > To verify it's returning the right results in the format you expect, you can 
# > test any Model from it's *Overview* page.
#
# If you entered an *Example Input* before, it will be the default input here, 
# though you can enter your own.

## Using the Model
#
# > The *Overview* page also provides sample `curl` or Python commands for calling your Model API.
# > You can adapt these samples for other code that will call this API.
#
# This is also where you can find the full endpoint to share with other developers 
# and data scientists.
#
# **Note:** for security, you can specify 
# [Model API Keys](https://docs.cloudera.com/machine-learning/cloud/models/topics/ml-model-api-key-for-models.html) 
# to add authentication.

## Limitations
#
# Models do have a few limitations that are important to know:
# - re-deploying or re-building Models results in Model downtime (usually brief)
# - re-starting CML does not automatically restart active Models
# - Model logs and statistics are only preserved so long as the individual replica is active
#
# A current list of known limitations are 
# [documented here](https://docs.cloudera.com/machine-learning/cloud/models/topics/ml-models-known-issues-and-limitations.html).

from datetime import datetime
import sys
import torch
import torch.nn as nn

class autoencoder(nn.Module):
    def __init__(self,num_features):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 15),
            nn.ReLU(True),
            nn.Linear(15, 7))
        self.decoder = nn.Sequential(
            nn.Linear(7, 15),
            nn.ReLU(True),
            nn.Linear(15, num_features),
            nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

num_features=30
split_point=-1.15

import joblib
scaler=joblib.load('model/cc_scaler.pkl')

model = autoencoder(num_features)
model.load_state_dict(torch.load('model/creditcard-fraud.model'))
model.eval()

def predict(args):
    with torch.no_grad():
        inp=[args['time']]+args['v']+[args['amount']]
        inp=scaler.transform([inp])
        inp=torch.tensor(inp, dtype=torch.float32)
        outp=model(inp)
        loss=torch.sum((inp-outp)**2,dim=1).sqrt().log()
        return {"result" : loss.item()>split_point}

#args = {"v": [-1.3598071336738,-0.0727811733098497,2.53634673796914,1.37815522427443,-0.338320769942518,0.462387777762292,0.239598554061257,0.0986979012610507,0.363786969611213,0.0907941719789316,-0.551599533260813,-0.617800855762348,-0.991389847235408,-0.311169353699879,1.46817697209427,-0.470400525259478,0.207971241929242,0.0257905801985591,0.403992960255733,0.251412098239705,-0.018306777944153,0.277837575558899,-0.110473910188767,0.0669280749146731,0.128539358273528,-0.189114843888824,0.133558376740387,-0.0210530534538215], "time": 0, "amount": 149.62}