# PUT ALL YOUR NON-FUNCTION CODE OVER HERE
# EGS: IMPORT STATEMENTS, LOADING PICKLE FILES / MODELS, DATASET/JSON PROCESSING, ETC.

# REMEMBER TO PLACE YOUR FILES (.PICKLE ETC.) IN THE FOLDER ABOVE THIS ONE I.E.
# IN THE SAME FOLDER AS RUN_APP.PY

import pandas as pd
import numpy as np
import pprint
from collections import namedtuple
import pickle
from sklearn.preprocessing import StandardScaler
from string import punctuation
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from nltk.stem.snowball import SnowballStemmer
pp = pprint.PrettyPrinter(indent=4)

scaler = pickle.load(open('genre_feature_scaler.pickle', 'rb'))
genres_cl = pickle.load(open('genres_voting_classifier.pickle', 'rb'))
genres_cl2 = pickle.load(open('genres_voting_classifier2.pickle', 'rb'))
genres_cl3 = pickle.load(open('genres_voting_classifier3.pickle', 'rb'))
genres_svc_ovo = pickle.load(open('genres_svc_ovo.pickle', 'rb'))
genres_svc_ovr = pickle.load(open('genres_svc_ovr.pickle', 'rb'))
genres_rfc1 = pickle.load(open('genres_rfc1.pickle', 'rb'))


mlb = pickle.load(open('mlb.pickle', 'rb'))
moods_vect1 = pickle.load(open('tfidf_vectorizer2.pickle', 'rb'))
moods_cl1 = pickle.load(open('moods_chain_rfc3.pickle', 'rb'))
moods_vect2 = pickle.load(open('bow_count_vect.pickle', 'rb'))
moods_cl2 = pickle.load(open('moods_chain_rfc5.pickle', 'rb'))
moods_vect3 = pickle.load(open('tfidf_vectorizer_en.pickle', 'rb'))
moods_cl3 = pickle.load(open('moods_chain_rfc_en1.pickle', 'rb'))

match_data = pickle.load(open('lookup_data.pickle', 'rb'))


def clean_text(raw_text):
    clean_words = []   
    # 1. Convert to lower case
    raw_text = raw_text.lower() 
    # 2. Remove punctuation
    translator = str.maketrans('', '', punctuation)
    raw_text = raw_text.translate(translator)
    split_words = raw_text.split() 
    # 3 & 4. Remove common words and stem words
    stemmer = SnowballStemmer('english')
    for word in split_words:
        if word not in ENGLISH_STOP_WORDS:
            stemmed_word = stemmer.stem(word)
            clean_words.append(stemmed_word)         
    return ' '.join(clean_words)



# THIS IS YOUR MAIN FUNCTION!
def recommend_similar_songs(audio_features, lyrics_features=""):
    # Genre
    features = np.asarray(audio_features)
    features = features.reshape(1,-1)
    test_song_scaled = scaler.transform(features)
    genre = genres_cl3.predict(test_song_scaled)
    print(genre[0])
    
    print(genres_svc_ovo.predict(test_song_scaled)[0])
    print(genres_svc_ovr.predict(test_song_scaled)[0])  
    print(genres_rfc1.predict(test_song_scaled)[0])
    
    if lyrics_features == "":
        genre_data = match_data[match_data['genres'].str.contains(genre[0])]
        final_data = genre_data
    else:
        # Moods
        # Clean text
        clean_lyrics = clean_text(lyrics_features)
        lyrics_list = []
        lyrics_list.append(clean_lyrics)
        # CL 1
        lyrics1 = moods_vect1.transform(lyrics_list)
        moods_proba1 = moods_cl1.predict_proba(lyrics1)
        # CL 2
        lyrics2 = moods_vect2.transform(lyrics_list)
        moods_proba2 = moods_cl2.predict_proba(lyrics2)
        # CL 3
        lyrics3 = moods_vect3.transform(lyrics_list)
        moods_proba3 = moods_cl3.predict_proba(lyrics3)
        # Put all in a mood list
        all_moods_list = pd.DataFrame(
        {'mood': mlb.classes_.tolist(),
         'cl1': moods_proba1.tolist()[0],
         'cl2': moods_proba2.tolist()[0],
         'cl3': moods_proba3.tolist()[0]
        })  
        # Calculate the max score
        all_moods_list['max'] = all_moods_list.max(axis=1)
        all_moods_list.sort_values('max', axis=0, ascending=False, inplace=True)
        all_moods_list.reset_index(drop=True, inplace=True)

        # Select relevant moods
        threshold = 0.5
        if all_moods_list.loc[0,('max')] >= threshold:
            mood_1 = all_moods_list['mood'][0]
        else:
            mood_1 = ""
        if all_moods_list.loc[1,('max')] >= threshold:
            mood_2 = all_moods_list['mood'][1]
        else:
            mood_2 = ""
        if all_moods_list.loc[2,('max')] >= threshold:
            mood_3 = all_moods_list['mood'][2]
        else:
            mood_3 = ""

        print(mood_1)
        print(mood_2)
        print(mood_3)

        # Match songs
        genre_data = match_data[match_data['genres'].str.contains(genre[0])]
        if mood_1 != "":
            filtered_data1 = genre_data[genre_data['moods'].str.contains(mood_1)]
            if mood_2 != "":
                filtered_data2 = filtered_data1[filtered_data1['moods'].str.contains(mood_2)]
                if mood_3 != "":
                    filtered_data3 = filtered_data2[filtered_data2['moods'].str.contains(mood_3)]
                    if len(filtered_data3) >= 10:
                        final_data = filtered_data3
                    else:
                        if len(filtered_data2) >= 10:
                            final_data = filtered_data2
                        else:
                            if len(filtered_data1) >= 10:
                                final_data = filtered_data1
                            else:
                                final_data = genre_data
                else:
                    if len(filtered_data2) >= 10:
                            final_data = filtered_data2
                    else:
                        if len(filtered_data1) >= 10:
                            final_data = filtered_data1
                        else:
                            final_data = genre_data
            else:
                if len(filtered_data1) >= 10:
                    final_data = filtered_data1
                else:
                    final_data = genre_data
        else:
            final_data = genre_data
                    
            
    playlist_df = final_data.sample(n=10)
    print(playlist_df.iloc[0,2])
    
    #Songs tuples
    Song = namedtuple("Song", ["artist", "title"])
    song_1 = Song(artist=playlist_df.iloc[0,2], title=playlist_df.iloc[0,1])
    song_2 = Song(artist=playlist_df.iloc[1,2], title=playlist_df.iloc[1,1])
    song_3 = Song(artist=playlist_df.iloc[2,2], title=playlist_df.iloc[2,1])
    song_4 = Song(artist=playlist_df.iloc[3,2], title=playlist_df.iloc[3,1])
    song_5 = Song(artist=playlist_df.iloc[4,2], title=playlist_df.iloc[4,1])
    song_6 = Song(artist=playlist_df.iloc[5,2], title=playlist_df.iloc[5,1])
    song_7 = Song(artist=playlist_df.iloc[6,2], title=playlist_df.iloc[6,1])
    song_8 = Song(artist=playlist_df.iloc[7,2], title=playlist_df.iloc[7,1])
    song_9 = Song(artist=playlist_df.iloc[8,2], title=playlist_df.iloc[8,1])
    song_10 = Song(artist=playlist_df.iloc[9,2], title=playlist_df.iloc[9,1])
        
    final_result_dictionary = dict(playlist=[song_1._asdict(),song_2._asdict(), song_3._asdict(), song_4._asdict(), song_5._asdict(), song_6._asdict(), song_7._asdict(), song_8._asdict(), song_9._asdict(), song_10._asdict()])
    
    # Genre result
    final_result_dictionary['genre'] = genre[0]
    
    # Moods tuples
    Mood = namedtuple("Mood", ["description", "probability"])
    if mood_1 != "":
        top_mood_1 = Mood(description=mood_1, probability=all_moods_list['max'][0])
        if mood_2 != "":
            top_mood_2 = Mood(description=mood_2, probability=all_moods_list['max'][1])
            if mood_3 != "":
                top_mood_3 = Mood(description=mood_3, probability=all_moods_list['max'][2])
                final_result_dictionary['mood'] = [top_mood_1, top_mood_2, top_mood_3]
            else:
                final_result_dictionary['mood'] = [top_mood_1, top_mood_2]
        else:
            final_result_dictionary['mood'] = [top_mood_1]
    else:
        final_result_dictionary['mood'] = 'No matching mood available'
        
    pp.pprint(final_result_dictionary)
    return final_result_dictionary



# THIS FUNCTION CONVERTS THE AUDIO FEATURES INTO A LIST BEFORE SENDING THEM TO
# recommend_similar_songs
def get_similar_songs(features, lyrics):
  print(features)
  print(lyrics)

  # features is a dict. convert it to a list using the same order as the assignments...
  audio_feature_headers = ['key', 'energy', 'liveness', 'tempo', 'speechiness', 'acousticness', 'instrumentalness', 'time_signature', 'duration', 'loudness', 'valence', 'danceability', 'mode', 'time_signature_confidence', 'tempo_confidence', 'key_confidence', 'mode_confidence']
  audio_features_list = []

  for audio_feature_name in audio_feature_headers:
      audio_features_list.append(features[audio_feature_name])

  # Provide the lyrics as is; a string

  return recommend_similar_songs(audio_features_list, lyrics)
