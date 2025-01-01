import streamlit as st
import pandas as pd
import numpy as np 
from bs4 import BeautifulSoup
import requests
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pickle

# Load the pipeline from the pipe.pkl file
with open('pipe.pkl', 'rb') as file:
    pipe = pickle.load(file)

# Load flag data
sf = pd.read_csv('flags_iso.csv')

st.header('T20 MATCH ANALYSIS')
o=20
o = st.number_input('Over No.(Not Greater Than Overs Played in 2nd Innings)') or 20
h = st.text_input('Enter the URL (ESPN CRICINFO > Select Match > Click On Overs):') or str('https://www.espncricinfo.com/series/icc-men-s-t20-world-cup-2024-1411166/australia-vs-india-51st-match-super-eights-group-1-1415751/match-overs-comparison')

if h == 'https://www.espncricinfo.com/series/icc-men-s-t20-world-cup-2024-1411166/australia-vs-india-51st-match-super-eights-group-1-1415751/match-overs-comparison':
    st.write('Enter Your URL')

r = requests.get(h)
b = BeautifulSoup(r.text, 'html')

venue = b.find(class_='ds-flex ds-items-center').text.split(',')[1]
list = []
list1 = []
list2 = []
list3 = []
list4 = []
list5 = []
list6 = []
list7 = []
list8 = []
list9 = []
list10 = []

elements = b.find_all(class_='ds-cursor-pointer ds-pt-1')
for i, element in enumerate(elements):
    if not element.text.split('/'):
        print(' ')
    else:
        if i % 2 != 0:
            list.append(element.text.split('/')[0])
            list1.append(element.text.split('/')[1].split('(')[0])

for i, element in enumerate(elements):
    if element.text.split('/') is None:
        print(' ')
    else:
        if i % 2 == 0:
            list8.append(element.text.split('/')[0])
            list9.append(i / 2 + 1)
            list10.append(element.text.split('/')[1].split('(')[0])

dict1 = {'inng1': list8, 'over': list9, 'wickets': list10}
df1 = pd.DataFrame(dict1)

for i in range(len(list)):
    list2.append(b.find_all(class_='ds-text-tight-s ds-font-regular ds-flex ds-justify-center ds-items-center ds-w-7 ds-h-7 ds-rounded-full ds-border ds-border-ui-stroke ds-bg-fill-content-prime')[i].text)
    list3.append(b.find(class_='ds-text-compact-m ds-text-typo ds-text-right ds-whitespace-nowrap').text.split('/')[0])
    list4.append(b.find_all('th', class_='ds-min-w-max')[1].text)
    list5.append(b.find_all('th', class_='ds-min-w-max')[2].text)
    list6.append(b.find(class_='ds-flex ds-items-center').text.split(',')[1])

if o == 20:
    list7.append(b.find(class_='ds-text-tight-s ds-font-medium ds-truncate ds-text-typo').text.split(' ')[0])

if o == 20:
    dict = {'batting_team': list5, 'bowling_team': list4, 'venue': list6, 'score': list, 'wickets': list1, 'over': list2, 'target': list3, 'winner': list7}
else:
    dict = {'batting_team': list5, 'bowling_team': list4, 'venue': list6, 'score': list, 'wickets': list1, 'over': list2, 'target': list3}
max_len = max(len(list5), len(list4), len(list6), len(list), len(list1), len(list2), len(list3))

list5.extend([None] * (max_len - len(list5)))
list4.extend([None] * (max_len - len(list4)))
list6.extend([None] * (max_len - len(list6)))
list.extend([None] * (max_len - len(list)))
list1.extend([None] * (max_len - len(list1)))
list2.extend([None] * (max_len - len(list2)))
list3.extend([None] * (max_len - len(list3)))

dict = {'batting_team': list5, 'bowling_team': list4, 'venue': list6, 'score': list, 'wickets': list1, 'over': list2, 'target': list3}
df = pd.DataFrame(dict)

df['score'] = df['score'].astype('int')
df1['inng1'] = df1['inng1'].astype('int')
df1['wickets'] = df1['wickets'].astype('int')
df1['previous_wickets'] = df1['wickets'].shift(1)
df1['previous_wickets'].loc[0] = 0
df1['wic'] = df1['wickets'] - df1['previous_wickets']
df1['over'] = df1['over'].astype('int')
df['over'] = df['over'].astype('int')
df['wickets'] = df['wickets'].astype('int')
df['previous_wickets'] = df['wickets'].shift(1)
df['previous_wickets'].loc[0] = 0
df['wic'] = df['wickets'] - df['previous_wickets']
df['wickets'] = df['wickets'].astype('int')
df['previous_score'] = df['score'].shift(1)
df['previous_score'].loc[0] = 0
df['runs_in_over'] = df['score'] - df['previous_score']
df['target'] = df['target'].astype('int')
df['runs_left'] = df['target'] - df['score']
df = df[df['score'] < df['target']]
df['crr'] = (df['score'] / df['over'])
df['rrr'] = ((df['target'] - df['score']) / (20 - df['over']))
df['balls_left'] = 120 - (df['over'] * 6)
df['runs'] = df['score'].diff()
df['last_five_runs'] = df['runs'].rolling(window=4).sum()
df['wickets_in_over'] = df['wickets'].diff()
df['last_five_wickets'] = df['wickets_in_over'].rolling(window=4).sum()
df = df.fillna(20)
df['match_id'] = 100001
#st.write(df)
neg_idx = df1[df1['inng1'] < 0].diff().index
if not neg_idx.empty:
    df1 = df1[:neg_idx[0]]
lf = df
lf = lf[:int(o)]

st.subheader('Scorecard')

o = int(o)
if o != 20:
    col1, col2 = st.columns([1, 1])

    with col1:
        bowling_team = df['bowling_team'].unique()[0]
        batting_team = df['batting_team'].unique()[0]

        # Get the URL for the bowling team
        bowling_team_url = sf[sf['Country'] == bowling_team]['URL']
        if not bowling_team_url.empty:
            # Display the bowling team flag and name in the same line
            col_bowling, col_bowling_name = st.columns([1, 3])  # Adjust proportions as needed
            with col_bowling:
                st.image(bowling_team_url.values[0], width=50)  # Adjust width as needed
            with col_bowling_name:
                st.write(f"**{bowling_team}**")

        # Get the URL for the batting team
        batting_team_url = sf[sf['Country'] == batting_team]['URL']
        if not batting_team_url.empty:
            # Display the batting team flag and name in the same line
            col_batting, col_batting_name = st.columns([1, 3])  # Adjust proportions as needed
            with col_batting:
                st.image(batting_team_url.values[0], width=50)  # Adjust width as needed
            with col_batting_name:
                st.write(f"**{batting_team}**")

    with col2:
        st.markdown("<div style='text-align: right;'>", unsafe_allow_html=True)  # Ensure left alignment
        st.write(str(df['target'].unique()[0]) + '/' + str(df1.iloc[-1, 2]))
        st.write('(' + str(df.iloc[o - 1, 5]) + '/' + '20)' + '    ' + str(df.iloc[o - 1, 3]) + '/' + str(df.iloc[o - 1, 4]))
        st.text('crr : ' + str(df.iloc[o - 1, 10].round(2)) + '  rrr : ' + str(df.iloc[o - 1, 11].round(2)))
        st.write(batting_team + ' Required ' + str(df.iloc[o - 1, 9]) + ' runs in ' + str(df.iloc[o - 1, 12]) + ' balls')
        st.markdown("</div>", unsafe_allow_html=True)

else:
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"{df['bowling_team'].unique()[0]}")
        st.write(f"{df['batting_team'].unique()[0]}")

    with col2:
        st.write(str(df1.iloc[-1, 0]) + '/' + str(df1.iloc[-1, 2]))
        st.write('(' + str(df.iloc[-1, 5]) + '/' + '20) ' + str(df.iloc[-1, 3]) + '/' + str(df.iloc[-1, 4]))
        if 'winner' in df.columns and not df['winner'].empty:
            winner = df['winner'].unique()
            if len(winner) > 0:
                st.write(winner[0] + ' Won')
import pickle
# Specify the path to your pickle file
pickle_file_path = 'pipe.pkl'

with open(pickle_file_path, 'rb') as file:
    pipe = pickle.load(file)
df['battingTeam_x']=df['batting_team']
df['bowlingTeam_x']=df['bowling_team']
df['runs_x_y']=df['target']
df['city_y']=df['venue']
if o==20:
    gf=df[['battingTeam_x','bowlingTeam_x','city_y','runs_left','balls_left','wickets','runs_x_y','crr','rrr','last_five_runs','match_id']]
else:
    gf=df[['battingTeam_x','bowlingTeam_x','city_y','runs_left','balls_left','wickets','runs_x_y','crr','rrr','last_five_runs','match_id']].iloc[:o]
df=df.iloc[:o]
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['over'], y=df['score'], mode='lines',marker_color='green',name=gf['battingTeam_x'].values[0]))
fig.add_trace(go.Scatter(x=df1['over'], y=df1['inng1'], mode='lines',marker_color='red',name=gf['bowlingTeam_x'].values[0]))
fig.update_layout(title='Score Comperision')

st.write(fig)    
fig = go.Figure()
df['wicket_in_over']=-df['wic']
fig.add_trace(go.Bar(x=df['over'], y=df['runs_in_over'],marker_color='blue',name='Runs'))
fig.add_trace(go.Bar(x=df['over'], y=df['wicket_in_over'],marker_color='red',name='Wickets'))
fig.update_layout(barmode='stack', title='Innings Progression')
st.write(fig)
def match_progression(x_df,Id,pipe):
    match = x_df[x_df['match_id'] ==Id]
    match = match[(match['balls_left']%6 == 0)]
    temp_df = match[['battingTeam_x','bowlingTeam_x','city_y','runs_left','balls_left','wickets','runs_x_y','crr','rrr','last_five_runs']].fillna(0)
    temp_df = temp_df[temp_df['balls_left'] != 0]
    if temp_df.empty:
        print("Error: Match is not Existed")
        a=1
        return None, None
    result = pipe.predict_proba(temp_df)
    temp_df['lose'] = np.round(result.T[0]*100,1)
    temp_df['win'] = np.round(result.T[1]*100,1)
    temp_df['end_of_over'] = range(1,temp_df.shape[0]+1)
    target = temp_df['runs_x_y'].values[0]
    print("Target-",target)
    temp_df = temp_df[['end_of_over','lose','win']]
    return temp_df,target
temp_df,target = match_progression(gf,100001,pipe)
#temp_df



import plotly.graph_objects as go

#fig = go.Figure()
#fig.add_trace(go.Scatter(x=temp_df['end_of_over'], y=temp_df['win'], mode='lines',name=gf['battingTeam_x'].values[0]))
#fig.add_trace(go.Scatter(x=temp_df['end_of_over'], y=temp_df['lose'], mode='lines',name=gf['bowlingTeam_x'].values[0]))
#fig.update_layout(title='Target-' + str(target))

#st.write(fig)
fig = go.Figure()

# Loop through the 'win' values to apply conditional logic
for win in temp_df['win'].values:
    if win >= 50:
        fig.add_trace(go.Scatter(
            x=temp_df['end_of_over'], 
            y=temp_df['win'], 
            mode='lines', 
            name="Win Probability",
            line={"color": "green" , "width": 2}
        ))
        fig.add_shape(
            type="line",
            x0=temp_df['end_of_over'].min(),
            x1=temp_df['end_of_over'].max(),
            y0=50,
            y1=50,
            line={"color": "red", "width": 1, "dash": "dash"},
        )
        fig.update_layout(
            title="Win Percentage Graph",
            xaxis_title="Over",
            yaxis={
                "range": [-10, 110],
                "tickvals": [-10,0,50, 100,110],
                "ticktext": [gf['bowlingTeam_x'].values[0], '100%',"50%", '100%',gf['battingTeam_x'].values[0]],
                "showgrid": False 
            },
            showlegend=False
        )
    else:
        # Yellow color for wins < 50
        # Add the trace with the yellow color for win probability
        fig.add_trace(go.Scatter(
            x=temp_df['end_of_over'], 
            y=temp_df['win'], 
            mode='lines', 
            name="Win Probability",
            line={"color": "red", "width": 2}
        ))
        # Add the shape for the dashed red line at 50% probability
        fig.add_shape(
            type="line",
            x0=temp_df['end_of_over'].min(),
            x1=temp_df['end_of_over'].max(),
            y0=50,
            y1=50,
            line={"color": "green", "width": 0.01, "dash": "dash"},
        )
        fig.update_layout(
            title="Win Percentage Graph",
            xaxis_title="Over",
            yaxis={
                "range": [-10, 110],
                "tickvals": [-10,0,50, 100,110],
                "ticktext": [gf['bowlingTeam_x'].values[0], '100%',"50%", '100%',gf['battingTeam_x'].values[0]],
                "showgrid": False 
            },
            showlegend=False
        )



st.write(fig)
import plotly.graph_objects as go

fig = go.Figure()

# Data for x (over) and y (win probabilities)
x = temp_df['end_of_over']
y = temp_df['win']

# Lambda function to segment data
get_segment = lambda condition: [v if condition(i) else None for i, v in enumerate(y)]

# Lambda function for trace names
get_name = lambda condition: gf['battingTeam_x'].values[0] if condition else gf['bowlingTeam_x'].values[0]

# Lambda for line attributes
get_line_attrs = lambda color, width: {"color": color, "width": width}

# Generate segments using the lambda function
above_threshold = get_segment(lambda i: y[i] >= 50)  # Values where win >= 50
below_threshold = get_segment(lambda i: y[i] < 50)   # Values where win < 50

# Add trace for win >= 50 (green line)
fig.add_trace(go.Scatter(
    x=x,
    y=above_threshold,
    mode='lines',
    name=get_name(True),  # Dynamic name for win >= 50
    line=get_line_attrs("green", 2)  # Dynamic line attributes
))

# Add trace for win < 50 (red line)
fig.add_trace(go.Scatter(
    x=x,
    y=below_threshold,
    mode='lines',
    name=get_name(False),  # Dynamic name for win < 50
    line=get_line_attrs("red", 2)  # Dynamic line attributes
))

# Add dashed line at 50% probability using lambda for shape attributes
get_shape_attrs = lambda color, width, dash: {"type": "line", "x0": x.min(), "x1": x.max(), "y0": 50, "y1": 50, "line": {"color": color, "width": width, "dash": dash}}
fig.add_shape(**get_shape_attrs("blue", 1, "dash"))

# Update layout
fig.update_layout(
    title="Win Percentage Graph",
    xaxis_title="Over",
    yaxis={
        "range": [-10, 110],
        "tickvals": [-10, 0, 50, 100, 110],
        "ticktext": [
            gf['bowlingTeam_x'].values[0],
            '0%',
            "50%",
            '100%',
            gf['battingTeam_x'].values[0]
        ],
        "showgrid": False
    },
    showlegend=True
)

st.write(fig)
import plotly.graph_objects as go

fig = go.Figure()
x, y = temp_df['end_of_over'], temp_df['win']

# Function to generate line segments based on condition
def get_segments(x, y):
    segments = []
    for i in range(len(y) - 1):
        if y[i] >= 50 and y[i + 1] >= 50:
            segments.append((x[i], x[i + 1], y[i], y[i + 1], 'green'))
        elif y[i] < 50 and y[i + 1] < 50:
            segments.append((x[i], x[i + 1], y[i], y[i + 1], 'red'))
        else:
            mid_x = (x[i] + x[i + 1]) / 2
            mid_y = 50
            if y[i] >= 50:
                segments.append((x[i], mid_x, y[i], mid_y, 'green'))
                segments.append((mid_x, x[i + 1], mid_y, y[i + 1], 'red'))
            else:
                segments.append((x[i], mid_x, y[i], mid_y, 'red'))
                segments.append((mid_x, x[i + 1], mid_y, y[i + 1], 'green'))
    return segments

# Get line segments
segments = get_segments(x, y)

# Add traces for each segment
for segment in segments:
    fig.add_trace(go.Scatter(
        x=[segment[0], segment[1]],
        y=[segment[2], segment[3]],
        mode='lines',
        line=dict(color=segment[4], width=2),
        showlegend=False
    ))

# Add dashed line at 50% probability
fig.add_shape(
    type="line",
    x0=x.min(),
    x1=x.max(),
    y0=50,
    y1=50,
    line=dict(color="blue", width=1, dash="dash")
)

# Update layout
fig.update_layout(
    title="Win Percentage Graph",
    xaxis_title="Over",
    yaxis=dict(
        range=[-10, 110],
        tickvals=[-10, 0, 50, 100, 110],
        ticktext=[
            gf['bowlingTeam_x'].values[0],
            '0%',
            "50%",
            '100%',
            gf['battingTeam_x'].values[0]
        ],
        showgrid=False
    ),
    showlegend=False
)

st.write(fig)








