SEED = 42
TEST_SIZE = 0.2

# ---------- data files ----------
SAMPLE_DIR = "data/sample"
FULL_DIR = "data/full"
IMDB_FILE = "imdb.csv"
RT_FILE = "rotten_tomatoes_movies.csv"

# Rows per genre per source when building the bundled sample
SAMPLE_PER_GENRE = 300

# ---------- column names in the raw Kaggle files ----------
# Change these if your download uses different headers.
IMDB_COLS = {"title": "movie_name", "year": "year", "genre": "genre", "text": "description"}
RT_COLS = {"title": "movie_title", "year": "original_release_date", "genre": "genres", "text": "movie_info"}

# ---------- genre mapping ----------
# Raw genre (lowercase) -> shared label used by both sources.
# Anything not listed here is ignored.
GENRE_MAP = {
    # IMDb names
    "action": "Action", "adventure": "Action",
    "animation": "Animation",
    "comedy": "Comedy",
    "drama": "Drama",
    "horror": "Horror",
    "romance": "Romance",
    "sci-fi": "SciFi_Fantasy", "fantasy": "SciFi_Fantasy",
    "thriller": "Thriller_Mystery", "mystery": "Thriller_Mystery", "crime": "Thriller_Mystery",
    # Rotten Tomatoes names
    "action & adventure": "Action",
    "science fiction & fantasy": "SciFi_Fantasy",
    "mystery & suspense": "Thriller_Mystery",
}

# Movies have several genres, and both sites list them alphabetically,
# so "first listed" would be biased. Instead pick the most specific one
# using this order. Comedy and Drama are catch-alls, so they come last.
GENRE_PRIORITY = [
    "Animation", "Horror", "SciFi_Fantasy", "Romance",
    "Action", "Thriller_Mystery", "Comedy", "Drama",
]

# ---------- model settings ----------
TFIDF = {"stop_words": "english", "max_features": 20000, "min_df": 2}
LOGREG = {"max_iter": 1000, "class_weight": "balanced"}
RANDOM_FOREST = {"n_estimators": 200, "n_jobs": -1}
