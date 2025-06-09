import praw
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Use read-only mode — no login needed
reddit = praw.Reddit(
    client_id="7OkPk72YmmFXj3Z4b3iO_Q",
    client_secret="H3KCapBh5AwnAoB950-w-0ufZ7zbIA",
    user_agent="duolingo_sentiment_analysis_for_marketing_assignment"
)

# Fetch posts about Duolingo
subreddit = reddit.subreddit("languagelearning")
posts = subreddit.search("Duolingo", limit=50)

# Extract comments
comments = []
for post in posts:
    post.comments.replace_more(limit=0)
    for comment in post.comments:
        comments.append(comment.body)

# Sentiment Analysis
analyzer = SentimentIntensityAnalyzer()
results = []
for text in comments:
    sentiment = analyzer.polarity_scores(text)
    results.append({
        "comment": text,
        "neg": sentiment["neg"],
        "neu": sentiment["neu"],
        "pos": sentiment["pos"],
        "compound": sentiment["compound"],
        "label": "Positive" if sentiment["compound"] >= 0.05 else
                 "Negative" if sentiment["compound"] <= -0.05 else "Neutral"
    })

# Save to CSV
df = pd.DataFrame(results)
df.to_csv("duolingo_sentiment.csv", index=False)
print("Done! Results saved to duolingo_sentiment.csv")
