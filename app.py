import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_lottie import st_lottie
import json

def Table(df):
    fig=go.Figure(go.Table( columnorder = [1,2,3],
          columnwidth = [20,48],
            header=dict(values=[' Title','Description', 'URL'],
                        line_color='black',font=dict(color='black',size= 19),height=40,
                        fill_color='#dd571c',#
                        align=['left','center']),
                cells=dict(values=[df['title'],df['description'],df['url']],
                       fill_color='#ffdac4',line_color='grey',
                           font=dict(color='black', family="Lato", size=16),
                       align='left')))
    fig.update_layout(height=500,width=1000, title ={'text': "Coursera Recommendation", 'font': {'size': 22}},title_x=0.5
                     )
    return st.plotly_chart(fig,use_container_width=True)
    


def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


    
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
            
def create_soup(x):
    return ' '.join(x['category-subject-area']) + ' ' + ' '.join(x['associated-university-institution-company']) + ' ' + x['syllabus'] 




df= pd.read_csv("./webautomation_coursera.csv")
df.loc[df.duplicated(['title']) == True, 'title'] = df["title"]+" ("+df['associated-university-institution-company']+")"

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])


cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


def get_recommendations(title, cosine_sim=cosine_sim):
   
    idx = indices[title]

   
    sim_scores = list(enumerate(cosine_sim[idx]))

   
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    
    sim_scores = sim_scores[1:6]

    
    course_indices = [i[0] for i in sim_scores]

    return df.iloc[course_indices]


indices = pd.Series(df.index, index=df['title']).drop_duplicates()

features = ['category-subject-area', 'associated-university-institution-company', 'syllabus']

for feature in features:
    df[feature] = df[feature].apply(clean_data)


count = CountVectorizer(stop_words='english')
df['soup'] = df.apply(create_soup, axis=1)
count_matrix = count.fit_transform(df['soup'])




cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

df = df.reset_index()
indices = pd.Series(df.index, index=df['title'])
movie_list = df['title'].values


####################################################################
#streamlit
##################################################################

st.header('Coursera Recommendation System')
#lottie_coding = load_lottiefile("./m4.json")
#st_lottie(
 #   lottie_coding,
  #  speed=1,
   # reverse=False,
    #loop=True,
    #quality="low",height=220
#)
selected_movie = st.selectbox(
    "Type or select a course from the dropdown",
    movie_list
)

if st.button('Coursera Recommendation'):
    recommended_movie_names = get_recommendations(selected_movie)
  
    #list_of_recommended_movie = recommended_movie_names.to_list()
   # st.write(recommended_movie_names[['title', 'description']])
    with st.container():
        Table(recommended_movie_names)
    
st.write('  '
         )
st.write(' ')

