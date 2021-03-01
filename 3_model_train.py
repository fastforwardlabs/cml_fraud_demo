## Part 3: Model Training

# This script is used to train the pytorch fraud model and also shows you how to use the 
# Jobs to run model training and the Experiments feature of CML to facilitate model 
# tuning.

# If you haven't yet, run through the initialization steps in the README file and Part 1. 
# In Part 1, the data is imported into the `default.cc_data` table in Hive. 
# All data accesses fetch from Hive.
#
# To simply train the model once, run this file in a workbench session.
# 
# There are 2 other ways of running the model training process
#
# ***Scheduled Jobs***
#
# The **[Jobs](https://docs.cloudera.com/machine-learning/cloud/jobs-pipelines/topics/ml-creating-a-job.html)**
# feature allows for adhoc, recurring and depend jobs to run specific scripts. To run this model 
# training process as a job, create a new job by going to the Project window and clicking _Jobs >
# New Job_ and entering the following settings:
# * **Name** : Train Mdoel
# * **Script** : 3_train_model.py
# * **Arguments** : _Leave blank_
# * **Kernel** : Python 3
# * **Schedule** : Manual
# * **Engine Profile** : 1 vCPU / 2 GiB 
# The rest can be left as is. Once the job has been created, click **Run** to start a manual 
# run for that job.

# ***Experiments***
#
# Training a model for use in production requires testing many combinations of model parameters
# and picking the best one based on one or more metrics.
# In order to do this in a *principled*, *reproducible* way, an Experiment executes model training code with **versioning** of the **project code**, **input parameters**, and **output artifacts**.
# This is a very useful feature for testing a large number of hyperparameters in parallel on elastic cloud resources.

# **[Experiments](https://docs.cloudera.com/machine-learning/cloud/experiments/topics/ml-running-an-experiment.html)**. 
# run immediately and are used for testing different parameters in a model training process. 
# In this instance it would be use for hyperparameter optimisation. To run an experiment, from the 
# Project window click Experiments > Run Experiment with the following settings.
# * **Script** : 3_train_models.py
# * **Arguments** : 256 0.01 100 _(these are the batch_size, lr and num_epochs parameters to be passed to the pytorch model)_
# * **Kernel** : Python 3
# * **Engine Profile** : 1 vCPU / 2 GiB

# Click **Start Run** and the expriment will be sheduled to build and run. Once the Run is 
# completed you can view the outputs that are tracked with the experiment using the 
# `cdsw.track_metrics` function. It's worth reading through the code to get a sense of what 
# all is going on.

## More Details on Running Experiments
### Requirements
# Experiments have a few requirements:
# - model training code in a `.py` script, not a notebook
# - `requirements.txt` file listing package dependencies
# - a `cdsw-build.sh` script containing code to install all dependencies
#
# These three components are provided for the fraud model as `3_train_models.py`, `requirements.txt`,
# and `cdsw-build.sh`, respectively.
# You can see that `cdsw-build.sh` simply installs packages from `requirements.txt`.
# The code in `3_train_models.py` is largely identical to the code in the last notebook.
# with a few differences.
# 
# The first difference from the last notebook is at the "Experiments options" section.
# When you set up a new Experiment, you can enter 
# [**command line arguments**](https://docs.python.org/3/library/sys.html#sys.argv) 
# in standard Python fashion.
# This will be where you enter the combination of model hyperparameters that you wish to test.
#
# The other difference is at the end of the script.
# Here, the `cdsw` package (available by default) provides 
# [two methods](https://docs.cloudera.com/machine-learning/cloud/experiments/topics/ml-tracking-metrics.html) 
# to let the user evaluate results.
#
# **`cdsw.track_metric`** stores a single value which can be viewed in the Experiments UI.
# Here we store two metrics and the filepath to the saved model.
#
# **`cdsw.track_file`** stores a file for later inspection.
# Here we store the saved model, but we could also have saved a report csv, plot, or any other 
# output file.
#    

from datetime import datetime
import cdsw
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# load the data

try:
  from pyspark.sql import SparkSession
  spark = SparkSession\
    .builder\
    .appName("PythonSQL")\
    .master("local[*]")\
    .getOrCreate()

  spark_df = spark.sql("SELECT * FROM default.cc_data")
  spark_df.printSchema()
  data = spark_df.toPandas()
except:
  data = pd.read_csv("/home/cdsw/data/creditcardfraud.zip")

feature_names=data.columns.values[:-1]
train_test_set = data[data.Class==0][feature_names]

# split into train set and test set

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(train_test_set, test_size=0.2, random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler().fit(train_set)
train_set=scaler.transform(train_set)
test_set=scaler.transform(test_set)

#save the scaler to use with the deployed model.
import joblib 
joblib.dump(scaler, 'model/cc_scaler.pkl') 

class autoencoder(nn.Module):
    def __init__(self,num_input):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_input, 15),
            nn.ReLU(True),
            nn.Linear(15, 7))
        self.decoder = nn.Sequential(
            nn.Linear(7, 15),
            nn.ReLU(True),
            nn.Linear(15, num_input),
            nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



### Experiments options
# If you are running this as an experiment, pass the cv, solver and max_iter values
# as arguments in that order. e.g. `256 0.01 100`.

if len (sys.argv) == 4:
  try:
    batch_size = int(sys.argv[1])
    lr = float(sys.argv[2])
    num_epochs = int(sys.argv[3])
  except:
    sys.exit("Invalid Arguments passed to Experiment")
else:
    batch_size = 256
    lr = 0.01 
    num_epochs = 100

# Define data loader
    
batch_size = 256
if torch.cuda.is_available():
  num_workers = 0
else:
  num_workers = 4

inputs = torch.tensor(train_set, dtype=torch.float32)
dataset = TensorDataset(inputs)
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

# Define loss function and optimizer

model = autoencoder(inputs.shape[1])
criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

if torch.cuda.is_available():
  model.to('cuda') 
  criterion.to('cuda')
  inputs.to('cuda')

# model training

for epoch in range(num_epochs):
    model.train()
    loss_sum=0.0; num=0
    for inputs1, in dataloader:
        if torch.cuda.is_available():
          inputs1 = inputs1.to('cuda')
        outputs = model(inputs1)
        loss = criterion(outputs, inputs1)
        loss_sum+=loss.item()
        num+=(inputs1.shape[0]*inputs1.shape[1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1)%5 == 0:
        print('{} epoch [{}/{}], loss:{:.6f}'#, test_set_loss:{:.6f}'
                .format(datetime.now(), epoch + 1, num_epochs, loss_sum/num))


# model evaluation

model.eval()

with torch.no_grad():
    test_set2 = data[data.Class==1][feature_names]
    test_set2=scaler.transform(test_set2)
    inputs2=torch.tensor(test_set2, dtype=torch.float32)
    if torch.cuda.is_available():
      inputs2 = inputs2.to('cuda')
    
    outputs2=model(inputs2)
    loss2=torch.sum((inputs2-outputs2)**2,dim=1).sqrt().log()
    loss2 = loss2.cpu()
    
    test_set1=test_set[np.random.choice(len(test_set),size=len(loss2),replace=False)]
    inputs1=torch.tensor(test_set1, dtype=torch.float32)
    if torch.cuda.is_available():
      inputs1 = inputs1.to('cuda')
    outputs1=model(inputs1)
    loss1=torch.sum((inputs1-outputs1)**2,dim=1).sqrt().log()
    loss1 = loss1.cpu()
    
    # pd.Series(loss1.numpy()).hist(bins=100)
    # pd.Series(loss2.numpy()).hist(bins=100)


def precision_rate(split_point):
    rate1=(loss1<split_point).sum().item()/float(len(loss1))
    rate2=(loss2>split_point).sum().item()/float(len(loss2))
    return (rate1+rate2)/2            

def find_split_point(start,end,start_precision,end_precision):
    print(start,'->',end)
    delta=(end-start)/4.0
    precision=[start_precision]
    precision+=[precision_rate(start+i*delta) for i in range(1,4)]
    precision+=[end_precision]

    i = 0 if sum(precision[0:3])>sum(precision[1:4]) else 1
    j = i if sum(precision[i:i+3])>sum(precision[2:5]) else 2

    if end-start>0.01:
        return find_split_point(start+j*delta,start+(j+2)*delta,precision[j],precision[j+2])
    else:
        return start+delta*np.argmax(precision)


(start,end)=(loss1.max().item(),loss2.min().item())
(start,end)=(start,end) if start<end else (end,start)
split_point=find_split_point(start,end,precision_rate(start),precision_rate(end))
print('\nSplit point:',split_point)

# Update the deployed model split point
import subprocess
subprocess.call(["sed", "-i",  's/split_point=.*/split_point=' + str(round(split_point,3)) + "/ ", "/home/cdsw/4_model_deploy.py"])


# model precision rate

precision1=(loss1<split_point).sum().item()/float(len(loss1))
precision2=(loss2>split_point).sum().item()/float(len(loss2))
print('Precision rate for normal cases:',precision1)
print('Precision rate for fraud cases:',precision2)
print('Overall precision:',(precision1+precision2)/2)

torch.save(model.state_dict(), 'model/creditcard-fraud.model')

# track experiment metrics
# If running as as experiment, this will track the metrics and add the model trained in this 
# training run to the experiment history.
cdsw.track_metric("split_point",round(split_point,2))
cdsw.track_metric("precision",round(((precision1+precision2)/2),2))
cdsw.track_file('model/creditcard-fraud.model')
cdsw.track_file('model/cc_scaler.pkl')



