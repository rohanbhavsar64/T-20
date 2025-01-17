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
url='https://static.cricinfo.com/rss/livescores.xml'
ru=requests.get(url)
import xml.etree.ElementTree as ET
root = ET.fromstring(ru.text)

# Find all item elements and extract their links
links = []
for item in root.findall('.//item'):
    link = item.find('link').text.strip()
    links.append(link)
#h=links[-5].replace('live-cricket-score','match-overs-comparison')
#st.write(h)
o=20
o = st.number_input('Over No.(Not Greater Than Overs Played in 2nd Innings)') or 20
h=st.text_input('URL')
if h is None:
    h='https://www.espncricinfo.com/series/big-bash-league-2024-25-1443056/melbourne-renegades-vs-melbourne-stars-32nd-match-1443088/match-overs-comparison'
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
df1['previous_score'] = df1['inng1'].shift(1)
df1['previous_score'].loc[0] = 0
df1['runs_in_over'] = df1['inng1'] - df1['previous_score']
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
m=df['target'].values[0]/5
df.fillna(m)
df['match_id'] = 100001
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

# Specify the path to your pickle file
df['battingTeam_x']=df['batting_team']
df['bowlingTeam_x']=df['bowling_team']
df['runs_x_y']=df['target']
df['city_y']=df['venue']
if o==20:
    gf=df[['battingTeam_x','bowlingTeam_x','city_y','runs_left','balls_left','wickets','runs_x_y','crr','rrr','last_five_runs','match_id']]
else:
    gf=df[['battingTeam_x','bowlingTeam_x','city_y','runs_left','balls_left','wickets','runs_x_y','crr','rrr','last_five_runs','match_id']].iloc[:o]
df=df.iloc[:o]
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
left, middle, right = st.columns(3)
if left.button("Score Compression", use_container_width=True):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['over'], y=df['score'], mode='lines',marker_color='grey',name=gf['battingTeam_x'].values[0],line={'width': 4}))
    fig.add_trace(go.Scatter(x=df1['over'], y=df1['inng1'], mode='lines',marker_color='red',name=gf['bowlingTeam_x'].values[0],line={'width': 4}))
    fig.update_layout(title='Score Comperision')
    st.write(fig) 
if middle.button("Innings Progression",use_container_width=True):
    fig = go.Figure()
    df['wicket_in_over']=-df['wic']
    fig.add_trace(go.Bar(x=df['over'], y=df['runs_in_over'],marker_color='blue',name=gf['battingTeam_x'].values[0]))
#fig.add_trace(go.Bar(x=df['over'], y=df['wicket_in_over'],marker_color='red',name='Wickets'))
    fig.add_trace(go.Bar(x=df1['over'], y=df1['runs_in_over'],marker_color='green',name=gf['bowlingTeam_x'].values[0]))
#fig.add_trace(go.Bar(x=df1['over'], y=df1['wic'],marker_color='red',name='Wickets',name=gf['bowlingTeam_x'].values[0]))
    fig.update_layout(title='Innings Progression')
    st.write(fig)
if right.button("WIN PERCENTAGE GRAPH",type="secondary",use_container_width=True):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=temp_df['end_of_over'], y=temp_df['win'], mode='lines',marker_color='purple',name=gf['battingTeam_x'].values[0],line={'width': 5}))
    fig.add_trace(go.Scatter(x=temp_df['end_of_over'], y=temp_df['lose'], mode='lines',marker_color='#FFD700',name=gf['bowlingTeam_x'].values[0],line={'width': 5}))
    fig.update_layout(title='WIN PERCENTAGE GRAPH')
    st.write(fig)



