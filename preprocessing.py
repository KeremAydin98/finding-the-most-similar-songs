import pandas as pd

# Reading the data
df_all = pd.read_csv("lyrics-data.csv")

# Drop all the columns except Lyrics and Song Name
df = df_all.loc[:, df_all.columns.intersection(['Lyric','SName','language'])]

# Filter only English songs
df = df[df["language"] == "en"]
df = df.drop("language", axis=1)