#df = pd.read_csv('Test_Time_Series.csv')
#df['Date'] = pd.to_datetime(df.Date,errors='coerce')
#df.index = df['Date']
#del df['Date']

app = dash.Dash()

app.layout = html.Div([

dash_table.DataTable(
    id='table',
    columns=[{"name": i, "id": i} for i in df.columns],
    data=df.to_dict("rows")
    )
])

@app.callback(
    dash.dependencies.Output('table', 'data')
  
def update_table():
    return df


if __name__ == '__main__':
    app.run_server(debug=False,port=int(os.environ['CDSW_APP_PORT']),host='127.0.0.1')



