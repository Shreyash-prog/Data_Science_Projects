### Spotify Playlist Creation 

The objective of this work is to use spotify playlist to create Songs Recommender and identify similar songs using context based embedding. We will be creating embeddings of songnames based on sequence they occur in playlist and using context vectors to create final algorithm

The USP of this work is that we are not using any property of songs i.e. lyrics, singer etc to get to a recommendation engine

### About Data : 
This dataset is based on the subset of users in the #nowplaying dataset who publish their #nowplaying tweets via Spotify. In principle, the dataset holds users, their playlists and the tracks contained in these playlists. 

The csv-file holding the dataset contains the following columns: 
"user_id", "artistname", "trackname", "playlistname"
, where
- user_id is a hash of the user's Spotify user name
- artistname is the name of the artist
- trackname is the title of the track and
- playlistname is the name of the playlist that contains this track.
The separator used is , each entry is enclosed by double quotes and the escape character used is \.
A description of the generation of the dataset and the dataset itself can be found in the following paper:
Pichl, Martin; Zangerle, Eva; Specht, Günther: "Towards a Context-Aware Music Recommendation Approach: What is Hidden in the Playlist Name?" in 15th IEEE International Conference on Data Mining Workshops (ICDM 2015), pp. 1360-1365, IEEE, Atlantic City, 2015.

Statistics About Data :
1. Number of Distinct Users : ~15K
2. Number of Distinct Artists Across All Users : 285,596
3. Number of Distinct SOngs Across All Users : 1,978,499

After looking into data we observed following :

1. Distribution of number of playlist per user is higly skewed (Right Skew). This observation aligns with general hypothesis that most users have few distinct playlist (<15) and almost none have playlist > 50

![playlistperuser](https://github.com/Ashwinikumar1/NLP-DL/blob/master/Spotify_Playlist_Creation_Using_Word2vec/Outputs/playlist_per_user.png)

2.  Distribution of number of distinct artist per user is higly skewed (Right Skew). This observation aligns with general hypothesis that most users listen to a few artist (<50) and almost none listen to more tha 500 distinct artist

![playlistperuser](https://github.com/Ashwinikumar1/NLP-DL/blob/master/Spotify_Playlist_Creation_Using_Word2vec/Outputs/songs_peruser.png)

also, if you want to know top artist, songs and playlist names by frequency of occurence please refer to following [location](https://github.com/Ashwinikumar1/NLP-DL/tree/master/Spotify_Playlist_Creation_Using_Word2vec/Outputs)

### Data Preparation :
 As we wanted to use context based embeddings, the first step of data preparation was to convert playlist data (Song Level) to playlist level.
 
 1. For every songs, convert the assign a numeric id. Store the hash map of song name to numeric id in dictionary and vice versa
 2. Aggregate the data at playlist level, and create a new columns which contains list of all songs in playlist. The list of all songs with numeric id is our input to word2vec model

Few other considerations:
1. Remove all playlist which have single songs
2. Remove all playlist which have more that 500 songs as they seems like they are not organised properly
3. While grouping we dont follow any order, as in playlist songs are played randomly
4. If we wanted more data we could have shuffled the playlist in differnt ways but for our case we had enough data

The code use for data prepocessing and eda is [code](https://github.com/Ashwinikumar1/NLP-DL/blob/master/Spotify_Playlist_Creation_Using_Word2vec/Spotify_Datasets_EDA_%26_prep.ipynb)

### Model Training :

We consider sequence of songs in a playlist as input to the word2vec model. We train our own embedding using skip gram model with context window of 4. 
The model is trained using Gensim library and en embedding of 200 Dimensions is created

### Algorithm 1 : Given a songs return top 5 similar songs
In this function we make use of cosine similarity between input songs vector and all other songs vector and return top 5 songs with highest cosine similarity. It is interesting to note that with this we get similar songs without using any information about songs i.e. singer, artist, lyrics etc

#### Input : similar_songs('kashmir',5)  (Searching Songs similar to kashmir led zepplin)

#### Output : Searching for songs similar to : kashmir
        Similar songs are as follow
        kashmir - live: o2 arena, london - december 10, 2007
        immigrant song - Led Zepplin
        keep talking - 2011 remastered version  - Pink Floyd
        karma police - Radiohead
        joker and the thief
 
 Results looks similar in terms of genre and taste. Out Embeddings are working as expected
 
 ### Algorithm 2 : Given list of songs create playlist
 We average the embedding vectors of input songs and find similar songs to the input vector
 
 #### Input : create_play_list(['hey you','time','hypnotised','fix you'],10)
 
 ### Ouput : Searching for songs similar to : ['hey you', 'time', 'hypnotised', 'fix you']
      Playlist based on your list is as follows
          hänsyn
          holy dread!
          hon har ett sätt - 1998 digital remaster
          high speed
          highway of endless dreams
          i am a man of constant sorrow - o brother, where art thou? soundtrack/with band
          holiday
          gap
          human
          i always was your girl
    
    
 #### Input 2 : create_play_list(['enter sandman','fade to black','kashmir'],15)
      Playlist based on your list is as follows
        fear of the dark - 1998 remastered version
        eye of the tiger
        for whom the bell tolls
        fear of the dark
        fairies wear boots
        feel good inc
        du hast
        even flow
        fuel
        entre dos tierras
        ett slag färgat rött
        everlong
        fade to black - instrumental version
        fortunate son
        estranged
 
#### Highlighting the Problems with current datasets and next steps for further better results
1. Training Data is not clean and has lot of similar songs with different names. We could try to restrict the version of songs to 1 or 2 max based on frequency for example SOngs : kashmir , kashmir - live: o2 arena, london - december 10, 2007

2.  Songs with similar names can be of different taste based on the artist names. We should create vocab by combining strings of tracknames with the artist names
 



 

