## Part 5: Application

# This script explains how to create and deploy Applications in CML.
# This feature allows data scientists to **get ML solutions in front of stakeholders quickly**,
# including business users who need results fast.
# This may be good for sharing a **highly customized dashboard**, a **monitoring tool**, or a **product mockup**.

# CML is agnostic regarding frameworks.
# [Flask](https://flask.palletsprojects.com/en/1.1.x/),
# [Dash](https://plotly.com/dash/),
# or even [Tornado](https://www.tornadoweb.org/en/stable/) apps will all work.
# R users will find it easy to deploy Shiny apps.

# If you haven't yet, run through the initialization steps in the README file. Do that
# now

# This file is provides a sample Dash app script, ready for deployment,
# which displays various sample data outputs using the Model API deployed in 
# Part 4

### Deploying the Application
#
# > Once you have written an app that is working as desired, including in a test Session,
# > it can be deployed using the *New Application* dialog in the *Applications* tab in CML.

# After accepting the dialog, CML will deploy the application then *assign a URL* to 
# the Application using the subdomain you chose.
# 
# *Note:* This does not requirement the `cdsw-build.sh* file as it doen now follows a 
# seperate build process to deploy an application.
# 

# To create an Application using our sample Dash app, perform the following.
# This is a special step for this particular app:
#
# In the deployed Model from step 4, go to *Model* > *Settings* in CML and make a note (i.e. copy) the 
#"**Access Key**". eg - `mqc8ypo...pmj056y`
#
# While you're there, **disable** the additional Model authentication feature by unticking **Enable Authentication**.
#
# **Note**: Disabling authentication is only necessary for this Application to work.
# Ordinarily, you may want to keep Authentication in place.
#
# Next, add that access key to `accesskey` variable below.
#
# `access_key = "mperto28a8xnul81g7w58xvy3qbplerw";
#
# Finally, go to the *Applications* section of the Project and select *New Application* with the following:
# * **Name**: Churn Analysis App
# * **Subdomain**: churn-app _(note: this needs to be unique, so if you've done this before, 
# pick a more random subdomain name)_
# * **Script**: 5_application.py
# * **Kernel**: Python 3
# * **Engine Profile**: 1 vCPU / 2 GiB Memory
#
# Accept the inputs, and in a few minutes the Application will be ready to use.
#
# At this point, you will be able to open the Fraud Detection App.
# Here, you can choose from four sample data points representing good and bad classifications.
# ![Application view](images/app_view.png)
#
# The functionality is quite basic and purely illustrative, but it should give you a sense of how straightforward it is
# to create interactive, usable applications and deploy them in CML. The first part of
# the code will get 4 samples, one for each kind required for [precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall)
# testing. 

# For a sample of the kinds of powerful Applications that are possible,
# see the [Dash gallery](https://dash-gallery.plotly.host/Portal/).
# A world of interactivity is open for you to develop.


import os.path
from shutil import copy
import logging
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import json
from pyspark.sql import SparkSession
import pandas as pd
import requests

spark = SparkSession\
    .builder\
    .appName("PythonSQL")\
    .master("local[*]")\
    .getOrCreate()

try:
    spark_df = spark.sql("SELECT * FROM default.cc_data")
    class1_df = spark_df.filter("Class == 1")
    class0_df = spark_df.filter("Class == 0").limit(2000)
    sample_df = class1_df.union(class0_df)
    data = sample_df.toPandas()
except:
    all_data = pd.read_csv("/home/cdsw/data/creditcardfraud.zip")
    class1_pdf = all_data[all_data.Class == 1]
    class0_pdf = all_data[all_data.Class == 0].iloc[:2000]
    data_ = class1_pdf.append(class0_pdf)
    
    
features = list(data.columns.values)
features.remove('Class')

access_key = "mperto28a8xnul81g7w58xvy3qbplerw";

# helper functions
model_host = os.getenv("CDSW_API_URL").split(
    ":")[0] + "://modelservice." + os.getenv("CDSW_DOMAIN") 

# helper functions
def get_prediction_from_model(d):
  r = requests.post(
    model_host + '/model', 
    data='{{"accessKey":"{}","request":{}}}'.format(access_key,json.dumps(reformat_sample_for_model(d))), 
    headers={'Content-Type': 'application/json'})
  return r.json()["response"]["result"]

def reformat_sample_for_model(d):
  d = {x: d[x] for x in d if x not in ['description','isFraud','predict']}
  v_list = []
  for i in range(28):
    v_list.append(float(d['V'+str(i+1)]))
  return {"time":float(d["Time"]),"amount":float(d["Amount"]),"v":v_list}
    
#### Get sample records.

data_sample = data.sample(frac=0.02, replace=False)

all_recs = []

for record in json.loads(data_sample.to_json(orient='records')):
  model_prediction = get_prediction_from_model(record)
  record.update({"prediction" : model_prediction})
  all_recs.append(record)

df_updated = pd.DataFrame(all_recs)

df = df_updated[['prediction','Class','Time','Amount','V1','V2','V3','V4','V5','V6','V7','V8']]

df['prediction'] = df['prediction'].replace({False: 0, True: 1})

## The DASH Application  

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

import dash_table

# HTML layout

app.layout = html.Div([    
  html.H1(children='CML Fraud Detection Prototype'),
  #html.Button('Update Table', id='submit-val', n_clicks=0),
  dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        style_data_conditional=[
           {
              'if': {'filter_query': '{Class} = 0 && {prediction} = 0',
                    'column_id': ['Class','prediction']
                },
                'backgroundColor': 'lightskyblue',
                'color': 'white'
            },
            {
              'if': {'filter_query': '{Class} = 1 && {prediction} = 1',
                    'column_id': ['Class','prediction']
                },
                'backgroundColor': 'crimson',
                'color': 'white'
            },
            {
              'if': {'filter_query': '{Class} != {prediction}',
                     'column_id': ['Class','prediction']
                },
                'backgroundColor': 'coral',
                'color': 'white'
            }

        ]

    )
  ])

@app.callback(
    Output('table', 'data'),
    [Input('submit-val','n_clicks')]
)
def update_output():
    data = df.to_dict('records')
    return data

# This reduces the the output to the console window
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
  

if __name__ == '__main__':
    app.run_server(debug=False,port=int(os.environ['CDSW_APP_PORT']),host='127.0.0.1')
