import numpy as np
import pandas as pd
import glob 
import rasterio as rio
import os
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
from dash import no_update
from skimage import io
import statsmodels as sm

# Create dataframe from picture
final = []
dsRes = glob.glob(os.path.join('e:/IBM/Examples/3', '*.tif'))
for f in dsRes:
    im = rio.open(f).read(1)
    imarray = np.array(im)
    final.append(imarray)

final = pd.DataFrame(
    # concatenate column vectors
    np.hstack([
        # first flatten, then convert row vectors to columns
        f.ravel().reshape(-1, 1)
        # for each array in your list
        for f in final
    ])
)

final.replace(0, np.nan, inplace=True)
final = final.dropna(axis = 0, how ='any')

headers = ["Wheat yield 20", "Corn yield 21","Ca","K","Mg","N", "Organik","P", "pH","S"]
headers = [str(integer) for integer in headers]

final.columns = headers

# Create a dash application
app = dash.Dash(__name__)

# REVIEW1: Clear the layout and do not display exception till callback gets executed
app.config.suppress_callback_exceptions = True

# List of years 
year_list = final.columns[0:2].values.tolist()
# List of elements
element_list = final.columns[2:].values.tolist()
# Prepear data
final['Wheat yield 20'] = final['Wheat yield 20']/33.6
final['Corn yield 21'] = final['Corn yield 21']/13.1
final['Ca'] = final['Ca']/9.7
final['K'] = final['K']/0.8
final['Mg'] = final['Mg']/58.4
final['N'] = final['N']/1.8
final['Organik'] = final['Organik']/56
final['P'] = final['P']/1.6
final['pH'] = final['pH']/27.4
final['S'] = final['S']/16.8
# Create new dataframe
df_elements = pd.DataFrame({'Elements' : element_list, 'Mean' : final[element_list].mean(), 'Total' : final[element_list].sum()})
df_yield = pd.DataFrame({'Year' : year_list, 'Mean' : final[year_list].mean(), 'Total' : final[year_list].sum()})
# Upload images
img = list()
for i in dsRes:
    img.append(io.imread(i))
img_fig = px.imshow(np.array(img), facet_col=0, labels={'facet_col':'f'})
for i, f in enumerate(headers):
    img_fig.layout.annotations[i]['text'] = f
# The average content of plant nutrients
bar_fig = px.bar(df_elements, x='Elements', y='Mean', color='Elements', title='The average content of plant nutrients')
# The total content of plant nutrients
pie_fig = px.pie(df_elements, values='Total', names='Elements', title='The total content of plant nutrients')

# Application layout
app.layout = html.Div(children=[ 
                                html.H1('The influence of nutrients on the yield of crops', style={'textAlign':'center', 'color':'#503D36', 'font-size': 24}),
                                # Create an outer division 
                                html.Div([
                                    # Add an division
                                    html.Div([
                                        # Create an division for adding dropdown helper text for report type
                                        html.Div([html.H2('Chose year and crop:', style={'margin-right': '2em'}),]),
                                        # Enter your code below. Make sure you have correct formatting.
                                        dcc.Dropdown(id='input-year', options=[{'label': i, 'value': i} for i in year_list],
                                                    placeholder='Select a year',
                                                    style={'width':'80%', 'padding':'3px', 'font-size': '20px', 'text-align-last' : 'center'}),],
                                        style={'display':'flex'}),  # Place them next to each other using the division style
                                    # Add next division
                                    html.Div([
                                       # Create an division for adding dropdown helper text for choosing year
                                        html.Div([html.H2('Choose nutrient:', style={'margin-right': '2em'})]),
                                        dcc.Dropdown(id='input-elements', options=[{'label': i, 'value': i} for i in element_list],
                                                     placeholder="Select a nutrient", style={'width':'80%', 'padding':'3px', 'font-size': '20px', 'text-align-last' : 'center'}),],
                                        style={'display': 'flex'}),]),
                                # Add images and graph
                                html.Div(dcc.Graph(figure=img_fig)),   
                                html.Div([
                                        html.Div(dcc.Graph(figure=bar_fig)),
                                        html.Div(dcc.Graph(figure=pie_fig))
                                        ], style={'display': 'flex'}),
                                html.Div([
                                        html.Div(dcc.Graph(id='plot3')),
                                        html.Div(dcc.Graph(id='plot4'))
                                ], style={'display': 'flex'}),
                                html.Div([
                                        html.Div(dcc.Graph(id='plot5'))], style={'display': 'flex'}),
                                ])

# Callback function definition
@app.callback([Output(component_id='plot3', component_property='figure'),
               Output(component_id='plot4', component_property='figure'),
               Output(component_id='plot5', component_property='figure')],
               [Input(component_id='input-year', component_property='value'),
                Input(component_id='input-elements', component_property='value')],
               # REVIEW4: Holding output state till user enters all the form information. In this case, it will be chart type and year
               [State("plot3", "figure"), State("plot4", "figure"), State("plot5", "figure")])

# Add computation to callback function and return graph
def get_graph(elements, year, children1, children2, children3):
    df = final.filter([(year), (elements)], axis=1).set_axis(['Year', 'Element'], axis=1, inplace=False)
    df1 = df.sort_values('Element')
   
    # Dependence of plant nutrients on productivity.
    line_fig = px.line(df1, x='Element', y='Year', title='Dependence of plant nutrients on productivity')
    # The relationship between plant nutrients and yield
    scatter_fig = px.scatter(df,  x='Element', y='Year', trendline="ols", trendline_options=dict(log_x=False), title="The relationship between plant nutrients and yield")

    df2 = df
    df2['Element'].mask(df2['Element'] <= df2['Element'].quantile(0.25),df2['Element'].quantile(0.25), inplace=True)
    df2['Element'].mask((df2['Element'] > df2['Element'].quantile(0.25)) & (df2['Element'] <= df2['Element'].quantile(0.75)),df2['Element'].mean(), inplace=True)
    df2['Element'].mask(df2['Element'] > df2['Element'].quantile(0.75),df2['Element'].quantile(0.75), inplace=True)
    viol_fig = px.violin(df2, x='Element', y='Year', box=True, violinmode='group', hover_data=df.columns)
    viol_fig.update_xaxes(type='category')
    # Return dcc.Graph component to the empty division
    return [line_fig, scatter_fig, viol_fig]
                    
# Run the app
if __name__ == '__main__':
    app.run_server()