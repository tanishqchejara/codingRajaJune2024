import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template_string
import requests

# Step 1: Data Collection
url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.data"
names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv(url, sep='\t', names=names)

# Step 2: Data Preprocessing
print("Data shape:", df.shape)
print("\nData info:")
df.info()
print("\nMissing values:")
print(df.isnull().sum())
print("\nRating distribution:")
print(df['rating'].value_counts().sort_index())

# Step 3: User-Item Matrix
user_item_matrix = df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

# Step 4: Collaborative Filtering
def cosine_similarity(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def get_similar_users(user_id, n=10):
    user_vector = user_item_matrix.loc[user_id].values
    similarities = []
    for other_user in user_item_matrix.index:
        if other_user != user_id:
            other_vector = user_item_matrix.loc[other_user].values
            similarity = cosine_similarity(user_vector, other_vector)
            similarities.append((other_user, similarity))
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:n]

# Step 5: Model Evaluation (simplified)
def evaluate_model():
    # In a real scenario, you'd split the data and do proper cross-validation
    print("Model evaluation would be performed here in a real scenario.")

# Step 6: Top-N Recommendations
def get_top_n_recommendations(user_id, n=5):
    similar_users = get_similar_users(user_id)
    user_ratings = user_item_matrix.loc[user_id]
    unrated_items = user_ratings[user_ratings == 0].index
    
    recommendations = {}
    for item in unrated_items:
        weighted_sum = 0
        similarity_sum = 0
        for similar_user, similarity in similar_users:
            if user_item_matrix.loc[similar_user, item] > 0:
                weighted_sum += similarity * user_item_matrix.loc[similar_user, item]
                similarity_sum += similarity
        if similarity_sum > 0:
            recommendations[item] = weighted_sum / similarity_sum
    
    top_n = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n]
    return top_n

# Fetch movie titles
movie_url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.item"
movies_df = pd.read_csv(movie_url, sep='|', encoding='latin-1', header=None, usecols=[0, 1], names=['item_id', 'title'])
movies_dict = dict(zip(movies_df.item_id, movies_df.title))

# Step 7: Interactive Interface
app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Movie Recommender</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        h1 { color: #333; }
        form { margin-bottom: 20px; }
        input[type="number"] { width: 100px; }
        button { background-color: #4CAF50; color: white; padding: 10px 15px; border: none; cursor: pointer; }
        button:hover { background-color: #45a049; }
        ul { list-style-type: none; padding: 0; }
        li { margin-bottom: 10px; }
    </style>
</head>
<body>
    <h1>Movie Recommender System</h1>
    <form id="recommender-form">
        <label for="user-id">Enter User ID (1-943):</label>
        <input type="number" id="user-id" name="user_id" min="1" max="943" required>
        <button type="submit">Get Recommendations</button>
    </form>
    <div id="recommendations"></div>

    <script>
        document.getElementById('recommender-form').addEventListener('submit', function(e) {
            e.preventDefault();
            const userId = document.getElementById('user-id').value;
            fetch('/recommend', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({user_id: parseInt(userId)}),
            })
            .then(response => response.json())
            .then(data => {
                const recommendationsDiv = document.getElementById('recommendations');
                recommendationsDiv.innerHTML = '<h2>Top 5 Movie Recommendations:</h2>';
                const ul = document.createElement('ul');
                data.forEach(movie => {
                    const li = document.createElement('li');
                    li.textContent = `${movie.title} (Estimated Rating: ${movie.rating.toFixed(2)})`;
                    ul.appendChild(li);
                });
                recommendationsDiv.appendChild(ul);
            })
            .catch((error) => {
                console.error('Error:', error);
            });
        });
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = request.json['user_id']
    recommendations = get_top_n_recommendations(user_id)
    result = [{"title": movies_dict[movie_id], "rating": rating} for movie_id, rating in recommendations]
    return jsonify(result)

if __name__ == '__main__':
    print("Starting the web application...")
    evaluate_model()
    app.run(debug=True)