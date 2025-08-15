import pandas as pd
import matplotlib.pyplot as plt

# Load results from sentiment step
df = pd.read_csv("app/sentiment_results.csv")

# Count each sentiment
sentiment_counts = df["sentiment"].value_counts()

# Plot
plt.figure(figsize=(6, 4))
sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'])

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Tweets")
plt.xticks(rotation=0)
plt.tight_layout()

# Show or save
plt.savefig("app/sentiment_chart.png")  # Save the chart
plt.show()  # Or just display it
