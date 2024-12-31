import streamlit as st
import pandas as pd
import numpy as np 
from bs4 import BeautifulSoup
import requests
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pickle

# Load flag data
sf = pd.read_csv('flags_iso.csv')

st.header('ODI MATCH ANALYSIS')
st.sidebar.header('Analysis')

selected_section = st.sidebar.radio('Select a Section:', ('Score Comparison', 'Innings Progression', 'Win Probability', 'Current Predictor'))

# Score Comparison
o = st.number_input('Over No.(Not Greater Than Overs Played in 2nd Innings)')
h = st.text_input('URL (ESPN CRICINFO >Select Match > Click On Overs') or 'https://www.espncricinfo.com/series/icc-men-s-t20-world-cup-2024-1411166/australia-vs-india-51st-match-super-eights-group-1-1415751/match-overs-comparison

if h == 'https://www.espncricinfo.com/series/icc-men-s-t20-world-cup-2024-1411166/australia-vs-india-51st-match-super-eights-group-1-1415751/match-overs-comparison':
    st.write('Enter Your URL')

url2 = h.replace('match-overs-comparison', 'live-cricket-score')
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
df['target'] = df['target'].astype('int')
df['runs_left'] = df['target'] - df['score']
df = df[df['score'] < df['target']]
df['crr'] = (df['score'] / df['over'])
df['rrr'] = ((df['target'] - df['score']) / (50 - df['over']))
df['balls_left'] = 300 - (df['over'] * 6)
df['runs'] = df['score'].diff()
df['last_10'] = df['runs'].rolling(window=10).sum()
df['wickets_in_over'] = df['wickets'].diff()
df['last_10_wicket'] = df['wickets_in_over'].rolling(window=10).sum()
df = df.fillna(20)
df['match_id'] = 100001

neg_idx = df1[df1['inng1'] < 0].diff().index
if not neg_idx.empty:
    df1 = df1[:neg_idx[0]]
lf = df
lf = lf[:int(o)]

st.subheader('Scorecard')

o = int(o)
if o != 50:
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
        st.write('(' + str(df.iloc[o - 1, 5]) + '/' + '50)' + '    ' + str(df.iloc[o - 1, 3]) + '/' + str(df.iloc[o - 1, 4]))
        st.text('crr : ' + str(df.iloc[o - 1, 8].round(2)) + '  rrr : ' + str(df.iloc[o - 1, 9].round(2)))
        st.write(batting_team + ' Required ' + str(df.iloc[o - 1, 7]) + ' runs in ' + str(df.iloc[o - 1, 10]) + ' balls')
        st.markdown("</div>", unsafe_allow_html=True)

else:
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"{df['bowling_team'].unique()[0]}")
        st.write(f"{df['batting_team'].unique()[0]}")

    with col2:
        st.write(str(df['target'].unique()[0]))
        st.write('(' + str(df.iloc[-1, 5]) + '/' + '50) ' + str(df.iloc[-1, 3]) + '/' + str(df.iloc[-1, 4]))

if 'winner' in df.columns and not df['winner'].empty:
    winner = df['winner'].unique()
    if len(winner) > 0:
        st.write(winner[0] + ' Won')
def match_progression(x_df,Id,pipe):
    match = x_df[x_df['match_id'] ==Id]
    match = match[(match['ball'] == 6)]
    temp_df = match[['battingTeam_x','bowlingTeam_x','city_y','runs_left','balls_left','wickets','runs_x_y','crr','rrr','winner','last_five_runs']].fillna(0)
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
    runs = list(temp_df['runs_left'].values)
    new_runs = runs[:]
    runs.insert(0,target)
    temp_df['runs_after_over'] = np.array(runs)[:-1] - np.array(new_runs)
    wickets = list(temp_df['wickets'].values)
    new_wickets = wickets[:]
    new_wickets.insert(0,10)
    wickets.append(0)
    w = np.array(wickets)
    nw = np.array(new_wickets)
    temp_df['wickets_in_over'] = (nw - w)[0:temp_df.shape[0]]
    
    print("Target-",target)
    temp_df = temp_df[['end_of_over','runs_after_over','wickets_in_over','lose','win']]
    return temp_df,target
