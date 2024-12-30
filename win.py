import streamlit as st
import pandas as pd
import numpy as np
df=pd.read_csv('2nd Innings T20.csv') 
first=pd.read_csv('1st Innings T20.csv')
df.drop(columns=[
     'tossWinner'
],inplace=True)
df['matchId']=df['matchId'].str.split('_').str[0].astype(int)
df['runs']=df['runs'].astype(int)
df['Score']=df.groupby('matchId').runs.cumsum()
df['over']=df['balls'].astype(str)
df['overs']=df['over'].str.split('.').str.get(0).astype(int)
df['ball']=df['over'].str.split('.').str.get(1).astype(int)
first['id']=first['matchId']
df['inning']=2
first['inning']=1
df['Id']=df['matchId']
first= first.groupby('matchId').sum()['runs'].reset_index().merge(first,on='matchId')
df= df.groupby('matchId').sum()['runs'].reset_index().merge(df,on='matchId')
df['balls_left']=120-(6*df['overs'])-df['ball']
df['crr']=(df['Score']*6)/(120-df['balls_left'])
groups = df.groupby('matchId')
match_ids =df['matchId'].unique()
st.write(df)
last_five = []

# Iterate over matchIds to calculate rolling sum for 'runs_y'
for id in match_ids:
    group = groups.get_group(id)
    
    # Apply the rolling sum for the 'runs_y' column
    # Drop NaN values after rolling operation or fill them
    rolling_runs = group.rolling(window=24)['runs_y'].sum().fillna(0).tolist()
    
    # Extend the results into the last_five list
    last_five.extend(rolling_runs)

# Add the rolling sums as a new column
df['last_five_runs'] = last_five

# Calculate the rolling sum for 'player_out'
last_five1 = []

for id in match_ids:
    group = groups.get_group(id)
    
    # Apply the rolling sum for the 'player_out' column
    rolling_wickets = group.rolling(window=24)['player_out'].sum().fillna(0).tolist()
    
    # Extend the results into the last_five1 list
    last_five1.extend(rolling_wickets)

# Add the rolling sums for 'player_out' as a new column
df['last_five_wicket'] = last_five1



first['inning']=1

st.write(first)
# Example: Splitting by a delimiter (e.g., a space or comma)
first['matchId'] = first['matchId'].str.split('_').str.get(0).astype(int)
match_df=df.merge(first,left_on='matchId',right_on='matchId')
st.write(match_df)
match_df['rrr']=((match_df['runs_x_y']-match_df['Score'])*6)/match_df['balls_left']
match_df['runs_left']=match_df['runs_x_y']-match_df['Score']
match_df=match_df[match_df['runs_left']>=0]
match_df['x1']=match_df['runs_x_x']-match_df['runs_x_y']
match_df['winner']=match_df['x1'].apply(lambda x:1 if x >= 0 else 0)
df1=match_df[['battingTeam_x','bowlingTeam_x','city_y','runs_left','balls_left','wickets','runs_x_y','crr','rrr','winner','last_five_runs']]
df1=df1.dropna()
x=df1.drop(columns='winner')
y=df1['winner']
st.write(match_df)
