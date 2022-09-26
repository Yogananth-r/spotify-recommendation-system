import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
sns.set()
import streamlit as st

data=pd.read_csv('spotify.csv')

dataframe = data.drop(columns=['id', 'name', 'artists', 'release_date', 'year'])

from sklearn.preprocessing import MinMaxScaler
datatypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
normalization = data.select_dtypes(include=datatypes)
for col in normalization.columns:
    MinMaxScaler(col)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10)
features = kmeans.fit_predict(normalization)
data['features'] = features
MinMaxScaler(data['features'])


class Spotify_Recommendation():
    def __init__(self, dataset):
        self.dataset = dataset
    def recommend(self, songs, amount=1):
        distance = []
        song = self.dataset[(self.dataset.name.str.lower() == songs.lower())].head(1).values[0]
        rec = self.dataset[self.dataset.name.str.lower() != songs.lower()]
        for songs in tqdm(rec.values):
            d = 0
            for col in np.arange(len(rec.columns)):
                if not col in [1, 6, 12, 14, 18]:
                    d = d + np.absolute(float(song[col]) - float(songs[col]))
            distance.append(d)
        rec['distance'] = distance
        rec = rec.sort_values('distance')
        columns = ['artists', 'name']
        return rec[columns][:amount]


recommendations = Spotify_Recommendation(data)


st.title("Spotify Recommendation")

@st.experimental_memo(suppress_st_warning=True)
def userentry():
    song=st.text_area("Enter Song Name:")
    if (st.button(label='Recommend')):
        if len(song)<1:
            st.write(" ")
        else:
            ex=song
            pred=recommendations.recommend(ex,10)
            st.write(pred)
userentry()