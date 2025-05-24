import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

@st.cache_data
def load_data():
    ratings_cols = ['userId', 'movieId', 'rating', 'timestamp']
    ratings = pd.read_csv("C:\\movie_recom\\u.data", sep='\t', names=ratings_cols, encoding='latin-1')

    movie_cols = ['movieId', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
                  'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
                  'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                  'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    movies = pd.read_csv("C:\\movie_recom\\u.item", sep='|', names=movie_cols, encoding='latin-1')

    genre_columns = movie_cols[5:]
    movies['genres'] = movies[genre_columns].apply(
        lambda row: ' '.join([genre for genre, val in zip(genre_columns, row) if val == 1]),
        axis=1
    )

    return movies[['movieId', 'title', 'genres']], ratings  

movies, ratings = load_data()
movie_data = pd.merge(ratings, movies, on='movieId')
user_movie_matrix = movie_data.pivot_table(index='userId', columns='title', values='rating').fillna(0)
collab_similarity = cosine_similarity(user_movie_matrix.T)
movie_index = user_movie_matrix.columns
index_to_movie = {i: title for i, title in enumerate(movie_index)}
movie_to_index = {title: i for i, title in enumerate(movie_index)}
count_vect = CountVectorizer()
genre_matrix = count_vect.fit_transform(movies['genres'])
genre_similarity = cosine_similarity(genre_matrix) 
def recommend_movies(title, top_n=5):
    if title not in movie_to_index:
        return ["Movie not found in database."]

    idx = movie_to_index[title]

    sim_scores_collab = collab_similarity[idx]

    genre_idx = movies[movies['title'] == title].index[0]
    sim_scores_genre = genre_similarity[genre_idx]

    hybrid_scores = sim_scores_collab + sim_scores_genre[:len(sim_scores_collab)]
    top_indices = sorted(range(len(hybrid_scores)), key=lambda i: hybrid_scores[i], reverse=True)
    top_movies = [index_to_movie[i] for i in top_indices if index_to_movie[i] != title][:top_n]

    return top_movies
st.title("🎬 Movie Recommendation System")
selected_movie = st.selectbox("Choose a movie you like:", user_movie_matrix.columns)
if st.button("Recommend"):
    recommendations = recommend_movies(selected_movie)
    st.subheader("Top 5 Recommendations:")
    for i, movie in enumerate(recommendations, 1):
        st.write(f"{i}. {movie}")
