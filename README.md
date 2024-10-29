# Customer Review Analysis for a Brunch Restaurant

This analysis provides insights into customer reviews of my favourite brunch restaurant located in central Madrid. By examining customer feedback, we aim to identify key areas of satisfaction and improvement opportunities to enhance the dining experience and strengthen the restaurant’s position in a competitive market.

In this readme provides along the document the python code used for processing the ununstructured data of reviews that allowed to get the different charts analyzed along the document. Some conclutions and recommendations from a business analysis are provided at the end.  

---

## Introduction
This report presents an analysis of customer reviews for a popular brunch restaurant in a central Madrid neighborhood. With a menu featuring breakfast and brunch favorites, the restaurant has garnered attention from both locals and tourists who have shared their experiences through online reviews. This analysis aims to identify key areas of customer satisfaction and potential improvements, which can help enhance the dining experience and reinforce the restaurant’s position in a competitive market.

## Motivation
The growing popularity of brunch spots in Madrid has led to an increase in competition, making customer satisfaction crucial for maintaining a unique and appealing identity. By analyzing customer feedback, this report seeks to leverage insights from unstructured data to pinpoint factors influencing customer perceptions. These findings will provide actionable steps for the restaurant to directly address areas of concern and further emphasize aspects that diners already value, with the goal of attracting new guests and fostering loyalty among regular customers.

---

```python
## 1. Import Libraries and Initial Configuration

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from collections import Counter

# Download stopwords if not already downloaded
nltk.download('stopwords')
```

```python
# Load the dataset
try:
    reviews_df = pd.read_csv("reviews.csv")
except FileNotFoundError:
    print("File 'reviews.csv' not found. Please make sure it is in the root directory.")

# Define language filter (set 'en' for English or 'es' for Spanish)
language_selected = 'en'

# Define stop words based on the selected language
if language_selected == 'en':
    stop_words = stopwords.words('english')
elif language_selected == 'es':
    stop_words = stopwords.words('spanish')

# Filter for selected language reviews
filtered_reviews_df = reviews_df[reviews_df['Language'] == language_selected]

# Convert 'Stay Date' column to datetime and drop NaNs in dates
filtered_reviews_df['Stay Date'] = pd.to_datetime(filtered_reviews_df['Stay Date'], errors='coerce')
filtered_reviews_df = filtered_reviews_df.dropna(subset=['Stay Date'])
filtered_reviews_df.head()
```

## Chart Analysis

### Chart 1: Reviews evolution

This charts shows the evolution and trends of customer reviews of my favourite brunch restaurant. 

![image_url_or_path](https://github.com/adrianrodriguez-io/sentiment-analysis-brunch-restaurant/blob/4721d507100ead85664281f0ca7be8bccadbf493/chrts/ld-reviewsevolution.png))

```python
# Extract year and month for grouping
filtered_reviews_df['YearMonth'] = filtered_reviews_df['Stay Date'].dt.to_period('M')

# Group by 'YearMonth' and 'Rating' to count occurrences
grouped_reviews = filtered_reviews_df.groupby(['YearMonth', 'Rating']).size().unstack(fill_value=0)

# Plot stacked bar chart for review evolution
fig, ax = plt.subplots(figsize=(14, 7))
grouped_reviews.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
ax.set_title('Review Evolution Over Time by Rating')
ax.set_xlabel('Experience Date (Year and Month)')
ax.set_ylabel('Number of Reviews')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### Chart 2: Word Cloud

![image_url_or_path](https://github.com/adrianrodriguez-io/sentiment-analysis-brunch-restaurant/blob/4721d507100ead85664281f0ca7be8bccadbf493/chrts/ld-wordscloud.png)

This chart highlights the most frequently mentioned words in reviews, both in positive and negative contexts.

- **Positive Words (Green)**: Words like "Pancakes," "good," "service," "food," "breakfast," "delicious," and "friendly" are prominent, indicating customer appreciation for these aspects. This reflects that the food (especially pancakes), service, and atmosphere are strong points of the restaurant.
  
- **Negative Words (Red)**: Words such as "one," "wait," "long," "horrible," and "took" suggest that wait times and service management cause dissatisfaction for some customers. Terms like "crowded," "wait," and "horrible" imply issues with organization or attention when the restaurant is full.

```python
# Define sentiment based on a rating threshold
filtered_reviews_df['Sentiment'] = np.where(filtered_reviews_df['Rating'] >= 4, 'Positive', 'Negative')

# Separate positive and negative reviews
positive_reviews = filtered_reviews_df[filtered_reviews_df['Sentiment'] == 'Positive']['Review Text']
negative_reviews = filtered_reviews_df[filtered_reviews_df['Sentiment'] == 'Negative']['Review Text']

# Custom function to calculate word frequencies
def calculate_word_frequencies(reviews, stop_words):
    word_counts = Counter()
    for review in reviews:
        words = review.lower().split()
        filtered_words = [word for word in words if word not in stop_words]
        word_counts.update(filtered_words)
    return word_counts

# Calculate word frequencies for positive and negative reviews
positive_word_counts = calculate_word_frequencies(positive_reviews, stop_words)
negative_word_counts = calculate_word_frequencies(negative_reviews, stop_words)

# Function to color words based on sentiment
def color_func(word, **kwargs):
    return "green" if positive_word_counts.get(word, 0) > negative_word_counts.get(word, 0) else "red"

# Combine positive and negative word frequencies
combined_word_dict = {word: positive_word_counts.get(word, 0) + negative_word_counts.get(word, 0) for word in set(positive_word_counts.keys()).union(set(negative_word_counts.keys()))}

# Generate WordCloud image
wordcloud = WordCloud(width=800, height=400, background_color='white', color_func=color_func, max_words=200).generate_from_frequencies(combined_word_dict)

# Plot WordCloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
```

### Chart 3: Sentiment Bar Chart

![image_url_or_path](https://github.com/adrianrodriguez-io/sentiment-analysis-brunch-restaurant/blob/4721d507100ead85664281f0ca7be8bccadbf493/chrts/ld-percmentions.png)

This chart shows the proportion of positive and negative mentions of various keywords in reviews.

- **Positive Mentions**: "Breakfast" has the highest proportion of positive mentions, nearly 30%, indicating that the breakfast experience is one of the most valued aspects. "Service" and "staff" are also positively mentioned, though some negative feedback is present. Keywords like "food," "pancakes," and "delicious" further confirm that food quality is generally appreciated.

- **Negative Mentions**: Keywords such as "one" and "took" have a high proportion of negative mentions, often related to complaints about wait times or specific issues. Words like "French," "time," and "quite" show mixed sentiment, which could suggest varying opinions on certain dishes or waiting times.

```python
# Calculate word percentages
total_reviews = len(filtered_reviews_df)
all_words = set(positive_word_counts.keys()).union(set(negative_word_counts.keys()))

word_percentages = pd.DataFrame({
    'word': list(all_words),
    'positive_percentage': [positive_word_counts.get(word, 0) / total_reviews * 100 for word in all_words],
    'negative_percentage': [negative_word_counts.get(word, 0) / total_reviews * 100 for word in all_words]
})

# Calculate total percentages and select top words
word_percentages['total_percentage'] = word_percentages['positive_percentage'] + word_percentages['negative_percentage']
top_words = word_percentages.nlargest(20, 'total_percentage').sort_values('total_percentage', ascending=True)

# Plot frequency bar chart for sentiment
fig_bar, ax_bar = plt.subplots(figsize=(10, 8))
bar_width = 0.4
y_positions = np.arange(len(top_words))

ax_bar.barh(y_positions, top_words['negative_percentage'], color='red', label='Negative Sentiment', height=bar_width)
ax_bar.barh(y_positions + bar_width, top_words['positive_percentage'], color='green', label='Positive Sentiment', height=bar_width)
ax_bar.set_yticks(y_positions + bar_width / 2)
ax_bar.set_yticklabels(top_words['word'])
ax_bar.set_xlabel('Percentage of Total Reviews Mentioning Word')
ax_bar.set_ylabel('Words')
ax_bar.set_title('Word Mentions by Sentiment')
ax_bar.legend()
plt.tight_layout()
plt.show()
```

### Chart 4: Radar Chart by Sentiment

![image_url_or_path](https://github.com/adrianrodriguez-io/sentiment-analysis-brunch-restaurant/blob/4721d507100ead85664281f0ca7be8bccadbf493/chrts/ld-radar.png)

The radar chart compares mentions of keywords like "coffee," "brunch," "place," "staff," "dishes," and "food" based on sentiment.

- **Positive Aspects**: "Food" and "place" have the most prominent positive area, showing that both the food and the restaurant’s ambiance are strengths. "Brunch" and "staff" also lean toward positive feedback, although "staff" has some negative mentions, indicating mixed experiences with the service.
  
- **Negative Aspects**: "Staff" and "dishes" have notable negative feedback, suggesting that while food is generally appreciated, there are specific items or service-related aspects that need attention.

```python
# Prepare data for radar chart
df_keywords = pd.DataFrame({
    'Keyword': top_words['word'],
    'PositiveSentiment': top_words['positive_percentage'],
    'NegativeSentiment': top_words['negative_percentage']
})

# Configure radar chart
labels = np.array(df_keywords['Keyword'])
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # Close the loop

fig_radar, ax_radar = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Positive sentiment values
positive_values = df_keywords['PositiveSentiment'].tolist()
positive_values += positive_values[:1]
ax_radar.plot(angles, positive_values, linewidth=2, linestyle='solid', label='Positive Sentiment')
ax_radar.fill(angles, positive_values, alpha=0.25)

# Negative sentiment values
negative_values = df_keywords['NegativeSentiment'].tolist()
negative_values += negative_values[:1]
ax_radar.plot(angles, negative_values, linewidth=2, linestyle='solid', label='Negative Sentiment', color='red')
ax_radar.fill(angles, negative_values, alpha=0.25, color='red')

# Configure labels
ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(labels)
plt.yticks([10, 20, 30, 40, 50], ["10%", "20%", "30%", "40%", "50%"], color="grey", size=7)
plt.ylim(0, max(max(positive_values), max(negative_values)) + 10)
plt.title('Percentage of Reviews Mentioning Keywords by Sentiment')
ax_radar.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.show()

```

---

## Conclusions

1. **Food Quality**: Customer feedback shows broad satisfaction with the food, especially for popular items like "pancakes" and "breakfast." This suggests that the restaurant meets expectations in terms of culinary experience, positioning it as a popular brunch destination.

2. **Service Experience**: While service is mostly well-rated, some inconsistencies are evident, with frequent complaints about wait times and service issues during peak hours. This points to an opportunity to enhance the consistency of service quality.

3. **Ambience and Comfort**: Customers enjoy the restaurant’s atmosphere, especially when it's less crowded. However, during peak hours, comfort appears to decrease due to the volume of customers, as indicated by terms like "crowded" in negative reviews. 

---

## Recommendations

1. **Optimize Wait Time Management**:
   - The restaurant could reduce wait times by implementing a reservation system and a queue management strategy, especially during peak hours. Reorganizing the table layout and adding more staff during busy times could further help streamline service.

2. **Staff Training for Consistent Service**:
   - Regular training for staff focused on enhancing customer service, especially during peak periods, could help improve the consistency of service. This could involve managing peak-time operations and ensuring that all customers receive the same level of quality in their dining experience.

3. **Promote Popular Dishes and Revise Less-Valued Ones**:
   - With items like "pancakes" and "breakfast" being highly popular, the restaurant could promote these dishes on social media or within its menu to attract more customers. Additionally, reviewing and potentially updating the preparation of dishes with mixed feedback, like certain French items, could enhance customer satisfaction.

4. **Enhance Comfort During Peak Times**:
   - Increasing the spacing between tables, adding waiting areas, or reconfiguring the layout could improve perceived comfort during busy periods. Limiting occupancy during peak hours may also help maintain a pleasant dining atmosphere.

---

## Final Remarks
This analysis offers valuable insights into customer satisfaction and areas for improvement, helping the restaurant maintain its favorable reputation. By addressing the highlighted issues and enhancing popular aspects of the dining experience, the restaurant can continue to thrive as a top brunch destination and improve customer loyalty.

