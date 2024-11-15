import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load and preprocess the dataset
df = pd.read_csv('YELP.Restaurants.csv')
df = df[["restaurant_name", "restaurant_tag", "rating", "price", "restaurant_neighborhood", "restaurant_address"]]
df['restaurant_tag'].fillna("No Tag", inplace=True)

# Convert price ranges to numerical values
price_mapping = {'$ ': 1, '$$ ': 2, '$$$ ': 3, '$$$$ ': 4}
df['price'] = df['price'].map(price_mapping)
df['price'].fillna(df['price'].mean(), inplace=True)
df['rating'].fillna(df['rating'].mean(), inplace=True)

# Prepare TF-IDF similarity based on tags
feature = df["restaurant_tag"].tolist()
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(feature)
tag_similarity = cosine_similarity(tfidf_matrix)

# Normalize rating and price
scaler = MinMaxScaler()
df[['rating', 'price']] = scaler.fit_transform(df[['rating', 'price']]) * 10
df['price'].fillna(df['price'].mean(), inplace=True)
df['rating'].fillna(df['rating'].mean(), inplace=True)

# Calculate cosine similarity for rating and price
rating_similarity = cosine_similarity(df[['rating']])
price_similarity = cosine_similarity(df[['price']])

tag_weight = 0.5
rating_weight = 0.3
price_weight = 0.2
combined_similarity = (
    tag_weight * tag_similarity +
    rating_weight * rating_similarity +
    price_weight * price_similarity
)

indices = pd.Series(df.index, index=df['restaurant_name']).drop_duplicates()

def get_recommendations(name):
    index = indices.get(name)
    if index is None:
        return None

    similarity_scores = list(enumerate(combined_similarity[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:11]

    restaurant_indices = [i[0] for i in similarity_scores]
    recommendations = df[['restaurant_name', 'rating', 'price', 'restaurant_neighborhood', 'restaurant_address']].iloc[restaurant_indices]
    return recommendations.to_dict(orient='records')

# Define a route to get recommendations
@app.route('/recommend', methods=['GET'])
def recommend():
    restaurant_name = request.args.get('name')
    recommendations = get_recommendations(restaurant_name)
    if recommendations is None:
        return jsonify({"error": "Restaurant not found"}), 404
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))
