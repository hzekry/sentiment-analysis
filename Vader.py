import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import seaborn as sns
from tqdm.notebook import tqdm

#read json file
df = pd.read_json('pet_related_reviews.json')
df = df.head(500)
#check nulls
print(df.isnull().sum())
 
df = df[['date','text', 'stars']]

df['text'] = df['text'].str.lower()
df['text'] = df['text'].str.replace('[^\w\s]','')
df['text'] = df['text'].str.replace('\d+','')
df['text'] =df['text'].str.strip()

df['tokens'] = df['text'].apply(word_tokenize)

stop_words = set(stopwords.words('english'))
df['tokens'] = df['tokens'].apply(lambda x: [word for word in x if word not in stop_words])

stemmer = PorterStemmer()
df['tokens'] = df['tokens'].apply(lambda x: [stemmer.stem(word) for word in x])
df['cleaned_text'] = df['tokens'].apply(lambda x: ' '.join(x))

sia = SentimentIntensityAnalyzer()
df['sentiment_score'] = df['cleaned_text'].apply(lambda x: sia.polarity_scores(x)['compound'])

res ={}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['text']
    myid = row['date']
    res[myid] = sia.polarity_scores(text)
      
vader = pd.DataFrame(res).T
vader = vader.reset_index().rename(columns={'index' :'date'})
vader = vader.merge(df, how='left')

print(vader)
ax= sns.barplot(data= vader, x='stars', y='compound')
ax.set_title('Compound score by review')
plt.show()

fig, axs = plt.subplots(1,3, figsize=(12,3))
sns.barplot(data=vader, x='stars', y='pos', ax= axs[0])
sns.barplot(data=vader, x='stars', y='neu', ax=axs[1])
sns.barplot(data=vader, x='stars', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[1].set_title('Negative')
plt.show()