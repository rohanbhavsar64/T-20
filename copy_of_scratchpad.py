# -*- coding: utf-8 -*-
"""Copy of scratchpad

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1UsXuwm-hbQ4fMetUyz1YSgT0W-kdP6oU
"""

import pandas as pd
import numpy as np
df1=pd.read_csv('df.csv')
df1=df1.dropna()
df1=df1[df1['rrr']<2000]
x=df1.drop(columns='winner')
y=df1['winner']
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.2,random_state=1)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()
trf = ColumnTransformer([
    ('trf',OneHotEncoder(sparse_output=False,handle_unknown = 'ignore'),['battingTeam_x','bowlingTeam_x','city_y'])],remainder='passthrough')
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
pipe=Pipeline(
    steps=[
        ('step1',trf),
        ('step2',LogisticRegression())
    ])
pipe.fit(xtrain,ytrain)

pipe.predict_proba(xtest)[0]

def match_progression(x_df,Id,pipe):
    match = x_df[x_df['match_id'] ==Id]
    match = match[(match['ball'] == 6)]
    temp_df = match[['battingTeam','bowlingTeam','city','runs_left','balls_left','wickets','runs_x_y','crr','rrr','winner','last_five_runs']].fillna(0)
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
temp_df,target = match_progression(match_df,1000,pipe)
temp_df
import plotly.graph_objects as go
fig = go.Figure()
wicket=fig.add_trace(go.Scatter(x=temp_df['end_of_over'], y=temp_df['wickets_in_over'], mode='markers', marker=dict(color='yellow')))
batting_team=fig.add_trace(go.Scatter(x=temp_df['end_of_over'], y=temp_df['win'], mode='lines', line=dict(color='#00a65a', width=3)))
bowling_team=fig.add_trace(go.Scatter(x=temp_df['end_of_over'], y=temp_df['lose'], mode='lines', line=dict(color='red', width=4)))
runs=fig.add_trace(go.Bar(x=temp_df['end_of_over'], y=temp_df['runs_after_over']))
fig.update_layout(title='Target-' + str(target))
fig