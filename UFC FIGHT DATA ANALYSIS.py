#!/usr/bin/env python
# coding: utf-8

# # UFC FIGHT DATA ANALYSIS

# In[33]:


import pandas as pd #helpful for manipulation of data
import numpy as np #deal with multidimensional error
import matplotlib.pyplot as plt
from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
import plotly.graph_objs as go
import seaborn as sns #it is a visualization library


# In[34]:


#extract fight type
def fight_type(X):
  weight_classes = ['Women\'s Strawweight', 'Women\'s Bantamweight', 
                  'Women\'s Featherweight', 'Women\'s Flyweight', 'Lightweight', 
                  'Welterweight', 'Middleweight','Light Heavyweight', 
                  'Heavyweight', 'Featherweight','Bantamweight', 'Flyweight', 'Open Weight']
  for Division in weight_classes:
        if Division in X:
            return Division
        if X == 'Catch Weight Bout' or 'Catchweight Bout':
            return 'Catch Weight'
        else:
            return 'Open Weight'

#determine age
def get_age(row):
    B_age = (row['date_year'] - row['B_year'])
    R_age = (row['date_year'] - row['R_year'])
    if np.isnan(B_age)!=True:
        B_age = B_age
    if np.isnan(R_age)!=True:
        R_age = R_age
    return pd.Series([B_age, R_age], index=['B_age', 'R_age'])

#determine number of rounds
def get_rounds(X):
    if X == 'No Time Limit':
        return 1
    else:
        return len(X.split('(')[1].replace(')', '').split('-'))

#change winer name to B or R, if NaN => Draw
def get_renamed_winner(row):
    if row['R_fighter'] == row['Winner']:
        return 'Red'
    elif row['B_fighter'] == row['Winner']:
        return 'Blue'
    elif row['Winner'] == 'Draw':
        return 'Draw'


# In[35]:


fighter_details = pd.read_csv('raw_fighter_details.csv')
fights_data = pd.read_csv('raw_total_fight_data.csv', sep=';')


# In[36]:


fighter_details.head()


# In[38]:


fighter_details.info()


# In[6]:


#fighter_details['Height'] = fighter_details['Height'].apply(to_cm)
#fighter_details['Reach'] = fighter_details['Reach'].apply(to_cm)
#fighter_details['Weight'] = fighter_details['Weight'].apply(to_kg)
#fighter_details['DOB'] = pd.to_datetime(fighter_details['DOB'])


# In[39]:


fighter_details.tail()


# In[40]:


fights_data.head()


# In[41]:


fights_data.info()


# In[42]:


fights_data['Fight_type'].value_counts()


# In[43]:


fights_data.columns.to_list


# In[44]:


# split attempted strikes and landed strikes
cols = ['R_SIG_STR.', 'B_SIG_STR.', 'R_TOTAL_STR.', 'B_TOTAL_STR.',
       'R_TD', 'B_TD', 'R_HEAD', 'B_HEAD', 'R_BODY','B_BODY', 'R_LEG', 'B_LEG', 
        'R_DISTANCE', 'B_DISTANCE', 'R_CLINCH','B_CLINCH', 'R_GROUND', 'B_GROUND']

attemp = '_attempted'
landed = '_landed'

for col in cols:
    fights_data[col+attemp] = fights_data[col].apply(lambda X: int(X.split('of')[1]))
    fights_data[col+landed] = fights_data[col].apply(lambda X: int(X.split('of')[0]))
    
fights_data.drop(cols, axis=1, inplace=True)
fights_data.head()


# In[45]:


fights_data.tail()


# In[46]:


# percentages to fractions

cols = ['R_SIG_STR_pct','B_SIG_STR_pct', 'R_TD_pct', 'B_TD_pct']

for col in cols:
    fights_data[col] = fights_data[col].apply(lambda X: float(X.replace('%', ''))/100)

fights_data.head()


# In[47]:


fights_data['Fight_type'].value_counts()


# In[48]:


fights_data['Fight_type'] = fights_data['Fight_type'].apply(fight_type)
fights_data['Fight_type'].value_counts()


# In[49]:


fights_data.info()


# In[50]:


fights_data['Winner'].isnull().sum() #5144 - 5061 #ITS SHOWS THAT HOW MANY NULL VALUE ARE THERE IN THIS BY SUM 


# In[66]:


fights_data['Winner'].fillna('Draw', inplace=True) #fill the null value with draw INPLACE=TRUE MEANS THAT IT WILL MAKE CHANGE AND DOESNOT RETURN ANY COPY


# In[52]:


fights_data['Winner'] = fights_data[['R_fighter', 'B_fighter', 'Winner']].apply(get_renamed_winner, axis=1) #apply is used to make changes
fights_data['Winner'].value_counts()


# In[53]:


df = fights_data.merge(fighter_details, left_on='R_fighter', right_on='fighter_name', how='left')
df.drop('fighter_name', axis=1, inplace=True)
df.rename(columns={'Height':'R_Height',
                          'Weight':'R_Weight',
                          'Reach':'R_Reach',
                          'Stance':'R_Stance',
                          'DOB':'R_DOB'}, 
                 inplace=True)


# In[54]:


df = df.merge(fighter_details, left_on='B_fighter', right_on='fighter_name', how='left')
df.drop('fighter_name', axis=1, inplace=True)
df.rename(columns={'Height':'B_Height',
                          'Weight':'B_Weight',
                          'Reach':'B_Reach',
                          'Stance':'B_Stance',
                          'DOB':'B_DOB'}, 
                 inplace=True)


# In[55]:


df.info()


# In[56]:


df['R_DOB'] = pd.to_datetime(df['R_DOB'])
df['B_DOB'] = pd.to_datetime(df['B_DOB'])
df['date'] = pd.to_datetime(df['date'])


# In[57]:


df['R_year'] = df['R_DOB'].apply(lambda x: x.year)
df['B_year'] = df['B_DOB'].apply(lambda x: x.year)
df['date_year'] = df['date'].apply(lambda x: x.year)
df[['B_age', 'R_age']]= df[['date_year', 'R_year', 'B_year']].apply(get_age, axis=1)
df.drop(['R_DOB', 'B_DOB','date_year','R_year','B_year'], axis=1, inplace=True)


# In[58]:


df['country'] = df['location'].apply(lambda x : x.split(',')[-1])


# In[59]:


from plotly.io import write_image
values = df.Winner.value_counts()
labels = values.index
colors = ['red', 'blue', 'green']
trace = go.Pie(labels=labels, 
               values=values,
                marker=dict(colors=colors) 
              )
layout = go.Layout(title='Winner Distribution BY PIE CHART')
fig = go.Figure(data=trace, layout=layout)
iplot(fig)
# fig.write_image("fig1.jpeg")


# In[60]:


df['R_age'] = df['R_age'].fillna(df['R_age'].median())
df['B_age'] = df['B_age'].fillna(df['B_age'].median())


# # TO CHECK THE YOUNGEST AND OLDEST FIGHTER

# In[61]:


df_rage = df[['R_fighter','R_age']].copy()
df_rage = df_rage.rename(columns={'R_fighter' : 'fighter', 'R_age' : 'age'})
df_bage = df[['B_fighter', 'B_age']].copy()
df_bage = df_bage.rename(columns={'B_fighter' : 'fighter', 'B_age' : 'age'})
df_aux = df_rage.append(df_bage, sort=False, ignore_index=True)
df_aux.drop_duplicates(subset='fighter', keep='first', inplace=True)

print("oldest fighter: ",df_aux.groupby('age').max().tail(1))
print("youngest fighter: ",df_aux.groupby('age').min().head(1))


# In[62]:


df['year'] = df['date'].apply(lambda x : int(str(x).split('-')[0]))


# In[63]:


values = df['year'].value_counts().sort_values(ascending=False)
clrs = ['b' if (x < max(values)) else 'r' for x in values ]


# # THE AGE OF R_FIGHTER AND B_FIGHTER

# In[64]:


plt.rcParams["xtick.labelsize"] = 10
plt.figure(figsize=(10,16))
f,ax=plt.subplots(1,2,figsize=(16,8))
df[df['Winner']=='Red']['R_age'].value_counts().plot.bar(ax=ax[0])

ax[0].set_title('R_age')
ax[0].set_ylabel('')
bar = df[df['Winner']=='Blue']['B_age'].value_counts().plot.bar(ax=ax[1])

ax[1].set_title('B_age')
plt.savefig('winners_age.png', dpi=300)
plt.show()


# # FIGHT WIN BY :

# In[65]:


values = df['win_by'].value_counts()
labels = values.index

plt.figure(figsize=(15,8))

sns.barplot(x=values,y=labels, palette='rocket')

plt.title('UFC Fight Win By')
plt.savefig('winby.png', dpi=300)

plt.show()


# In[ ]:




