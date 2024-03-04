# Movie Recommendation Function

# Before running this function, install the NLP model using "py -m spacy download en_core_web_md" in the cmd terminal
import pandas as pd
import spacy

# Load the movies.txt file and turn it into a dataset using pandas
file_path = r"C:\Users\user\OneDrive\Documents\Data Science Bootcamp\Task 20 - NLP - Semantic Similarity\movies.txt" 
# Update with local file path
movie_df = pd.read_csv(file_path, sep = ":", header = None)
movie_df.columns = ["Movie Name", "Description"]

# Function to recommend a different movie based on similarity to previously defined movie description 
def movie_recommendation(df, input_movie_descrip):
    # NLP processing of defined movie description
    nlp = spacy.load('en_core_web_md')
    check_movie = nlp(input_movie_descrip)

    # For loop generates similarity score for each description in the dataset and adds each score to a new column
    row = 0
    for item in df["Description"]:
          similarity = nlp(item).similarity(check_movie)
          df.loc[row, "Similarity Score"] = similarity
          row = row + 1

    # Sort similarity scores in decreasing order, with highest score being first
    df = df.sort_values(by = "Similarity Score", ascending = False)
    suggestion = df.iloc[0,0]
    
    return print("The most similar movie suggestion is " + suggestion)


user_movie_description = '''Will he save their world or destroy it? When the Hulk becomes 
                            too dangerous for the Earth, the Illuminati trick Hulk into
                            a shuttle and launch him into space to a planet where the 
                            Hulk can live in peace. Unfortunately, Hulk lands on the planet 
                            Sakaar where he is sold into slavery and trained as a gladiator'''


suggested_movie = movie_recommendation(movie_df, user_movie_description)

