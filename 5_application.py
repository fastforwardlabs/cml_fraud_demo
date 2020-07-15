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
# `access_key = "mva4xq8t7prim32i0y0258pof6tvu8ci";
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
import dill
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
    spark_df = spark.sql("SELECT * FROM default.cc_data limit 1000")
    data = spark_df.toPandas()
except:
    data = pd.read_csv("/home/cdsw/data/creditcardfraud.zip",nrows=1000)

features = list(data.columns.values)
features.remove('Class')

access_key = "mva4xq8t7prim32i0y0258pof6tvu8ci";

# helper functions
def get_prediction_from_model(d):
  r = requests.post(
    'http://modelservice.ml-2f4cffbb-91e.demo-aws.ylcu-atmi.cloudera.site/model', 
    data='{{"accessKey":"{}","request":{}}}'.format(access_key,json.dumps(reformat_sample_for_model(d))), 
    headers={'Content-Type': 'application/json'})
  return r.json()["response"]["result"]

def format_values(val):
    if type(val) is bool:
        return str(val)
    elif type(val) is str:
        return val
    elif val % 1 == 0:
        return str(int(val))
    else:
        return '{:.3f}'.format(val)

def format_sample(data, desc):
    out = data[features].to_dict()
    out['description'] = desc
    out['isFraud'] = data['Class'] == 1
    out['predict'] = data['prediction']
    return out

def make_table(args):
    table = html.Table(children=[])
    table.children.append(
        html.Tr(children=[html.Th(children=col) for col in features])
    )
    table.children.append(
        html.Tr(children=[
            html.Td(children=format_values(args[col])) for col in features
        ])
    )
    return table

def reformat_sample_for_model(d):
  d = {x: d[x] for x in d if x not in ['description','isFraud','predict']}
  v_list = []
  for i in range(28):
    v_list.append(float(d['V'+str(i+1)]))
  return {"time":float(d["Time"]),"amount":float(d["Amount"]),"v":v_list}
    
#df = pd.read_csv("/home/cdsw/data/creditcard.csv", nrows=1000)

#### Get 4 sample records, one of each kind,

all_recs = []

for record in json.loads(data.to_json(orient='records')):
  model_prediction = get_prediction_from_model(record)
  record.update({"prediction" : model_prediction})
  all_recs.append(record)

df_updated = pd.DataFrame(all_recs)

df_updated

features = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

sample1 = format_sample(df_updated.query('(Class == 0) & (prediction == False)').iloc[0], 'true inlier')
sample2 = format_sample(df_updated.query('(Class == 1) & (prediction == True)').iloc[0], 'true outlier')
sample3 = format_sample(df_updated.query('(Class == 0) & (prediction == True)').iloc[0], 'false outlier')
sample4 = format_sample(df_updated.query('(Class == 1) & (prediction == False)').iloc[0], 'false inlier')

samples = {'sample1': sample1,
           'sample2': sample2,
           'sample3': sample3,
           'sample4': sample4}

  
## The DASH Application  
  
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# HTML layout
app.layout = html.Div(children=[
    html.H1(children='CML Fraud Detection Template Project'),

    html.Div(children='''
        Pick a sample data point and see whether our anomaly detection model decides
        it's a fraudulent transaction
    ''',
    style={'margin-bottom': '40px'}),

    html.Label('Sample Data'),
    dcc.Dropdown(
        options=[
            {'label': 'Not Fraud', 'value': 'sample1'},
            {'label': 'Fraud', 'value': 'sample2'},
            {'label': 'Incorrectly labeled Fraud', 'value': 'sample3'},
            {'label': 'Incorrectly labeled Not Fraud', 'value': 'sample4'},
        ],
        value='sample1',
        id='switcher',
        searchable=False,
        clearable=False,
        style={'width': '50%'},
    ),

    html.Div(children=make_table(samples['sample1']), id='values'),

    html.Div(children=[
        html.Span(children='Fraud? ',
                  style={'font-weight': 'bold',
                         'display': 'inline-block',
                         'padding': '20px',
                         }),
        html.Span(children=str(samples['sample1']['isFraud']), id='true'),
    ]),

    html.Div(children=[
        html.Span(children='Predicted Fraud? ',
                  style={'font-weight': 'bold',
                         'display': 'inline-block',
                         'padding': '20px',
                         }),
        html.Span(children=str(get_prediction_from_model(samples['sample1'])), id='prediction'),
    ])
])


@app.callback(
    Output(component_id='true', component_property='children'),
    [Input(component_id='switcher', component_property='value')]
)
def update_true(input_value):
    return str(samples[input_value]['isFraud'])


@app.callback(
    Output(component_id='prediction', component_property='children'),
    [Input(component_id='switcher', component_property='value')]
)
def update_prediction(input_value):
    return str(get_prediction_from_model(samples[input_value]))


@app.callback(
    Output(component_id='values', component_property='children'),
    [Input(component_id='switcher', component_property='value')]
)
def update_table(input_value):
    return make_table(samples[input_value])

# This reduces the the output to the console window
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


if __name__ == '__main__':
    app.run_server(debug=False,port=int(os.environ['CDSW_APP_PORT']),host='127.0.0.1')






