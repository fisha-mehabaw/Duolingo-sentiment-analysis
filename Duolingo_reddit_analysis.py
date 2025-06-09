import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
from collections import Counter

nltk.download('vader_lexicon')


df = pd.read_csv("duolingo_sentiment.csv")
print(df.head())


sia = SentimentIntensityAnalyzer()

# Apply sentiment analysis
df['scores'] = df['comment'].apply(lambda comment: sia.polarity_scores(str(comment)))
df['compound'] = df['scores'].apply(lambda score_dict: score_dict['compound'])

# Classify sentiment
def classify_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['label'] = df['compound'].apply(classify_sentiment)
print(df['label'].value_counts())

# Plot pie chart
plt.figure(figsize=(6, 6))
df['label'].value_counts().plot.pie(autopct='%1.1f%%', startangle=140, colors=["#8BC34A", "#FFC107", "#F44336"])
plt.title('Sentiment Distribution of Duolingo Reddit Comments')
plt.ylabel('')
plt.tight_layout()
plt.show()

# Plot bar chart
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='label', order=['Positive', 'Neutral', 'Negative'], palette='pastel')
plt.title('Number of Comments per Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Show most extreme sentiments (top 3 each)
top_positive = df.sort_values(by='compound', ascending=False).head(3)
top_negative = df.sort_values(by='compound').head(3)

print("\nTop Positive Comments:")
print(top_positive[['comment', 'compound']])

print("\nTop Negative Comments:")
print(top_negative[['comment', 'compound']])


# Text preprocessing
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df['clean_comment'] = df['comment'].apply(preprocess)

# Vectorize the comments
vectorizer = CountVectorizer(stop_words='english', max_df=0.9, min_df=5)
dtm = vectorizer.fit_transform(df['clean_comment'])

# LDA for topic modeling
lda = LatentDirichletAllocation(n_components=20, random_state=42)
lda.fit(dtm)

# Display top words for each topic
print("\nTop Words Per Theme:")
for i, topic in enumerate(lda.components_):
    print(f"\nTheme {i+1}:")
    top_words = [vectorizer.get_feature_names_out()[j] for j in topic.argsort()[-10:]]
    print(", ".join(top_words))

# List of competitor brand names
competitors = [
    'babbel', 'busuu', 'memrise', 'rosetta stone',
    'lingq', 'mondly', 'hellotalk', 'tandem', 'lingvist',
    'coursera', 'edx', 'udemy',
    'pimsleur', 'drops', 'beelinguapp', 'italki', 'preply', 'speechling', 'openlanguage',
    'tatoeba', 'anki'
]


# check if any competitor is mentioned
df['comment_lower'] = df['comment'].str.lower()
competitor_mentions = df[df['comment_lower'].apply(lambda x: any(comp in x for comp in competitors))]

mentions_count = Counter()
for comment in competitor_mentions['comment_lower']:
    for comp in competitors:
        if comp in comment:
            mentions_count[comp] += 1

# Display frequency of each competitor
print("Competitor mention frequency:")
for comp, count in mentions_count.items():
    print(f"{comp.capitalize()}: {count}")

mentions_df = pd.DataFrame(mentions_count.items(), columns=['Competitor', 'Mentions'])
mentions_df['Competitor'] = mentions_df['Competitor'].str.capitalize()

plt.figure(figsize=(10,6))
barplot = sns.barplot(data=mentions_df, x='Competitor', y='Mentions', palette='viridis')

for p in barplot.patches:
    height = p.get_height()
    barplot.text(
        p.get_x() + p.get_width() / 2.,  
        height + 0.5,                    
        int(height),                    
        ha="center"
    )

plt.title('Competitor Mention Frequency')
plt.xlabel('Competitor')
plt.ylabel('Number of Mentions')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()