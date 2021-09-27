


from flask import Flask, request,render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')



@app.route('/',methods=['POST'])
def predict():
    data_movies = pd.read_csv('data.csv')
    data_movies = data_movies.drop('id',axis=1)

    # Content Based Movie Recommendation
    tfidf = TfidfVectorizer(stop_words='english')
    data_movies['overview'] = data_movies['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(data_movies['overview'])
    cosine = linear_kernel(tfidf_matrix)
    indices = pd.Series(data_movies.index,index=data_movies['title']).drop_duplicates()

    def get_recommendation(title,co_sine = cosine):
        idx = indices[title]
        sim_score = list(enumerate(co_sine[idx]))
        sim_score = sorted(sim_score,key=lambda x: x[1],reverse=True)
        sim_score = sim_score[1:11]
        movie_indices = [i[0] for i in sim_score]
        return data_movies['title'].iloc[movie_indices]

    query_movie=request.form['movie']

    movies = get_recommendation(query_movie)
    df = pd.DataFrame(movies)
    table = df.to_html(index=False)

    return render_template('home.html',table=table)


if __name__=="__main__":
    app.run(debug=True)


