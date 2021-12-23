### Objective
We will create a classification model to classify AG News Corpus. The corpus belongs to 4 different classes. The purpose of this project is to apply sequential model to text classification task


### Learning Objective 
To be able to apply every Sequential Model types to the prject and see performance

1. RNN using pretrained embddings - We will use Glove 100 D vector for this project
2. RNN using embedding trained as a part of model training
3. LSTM using pretrained embedding - We will use Glove 100D vector for this project
4. LSTM using embedding as a part of model training
5. GRU using embedding or pretrained embedding, whatever performed better

We will be using Pytorch for our projects
Data Description - https://www.kaggle.com/amananandrai/ag-news-classification-dataset

## Data Deatils
AG is a collection of more than 1 million news articles. News articles have been gathered from more than 2000 news sources by ComeToMyHead in more than 1 year of activity. ComeToMyHead is an academic news search engine which has been running since July, 2004. The dataset is provided by the academic comunity for research purposes in data mining (clustering, classification, etc), information retrieval (ranking, search, etc), xml, data compression, data streaming, and any other non-commercial activity. For more information, please refer to the link http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html .

The AG's news topic classification dataset is constructed by Xiang Zhang (xiang.zhang@nyu.edu) from the dataset above. It is used as a text classification benchmark in the following paper: Xiang Zhang, Junbo Zhao, Yann LeCun. Character-level Convolutional Networks for Text Classification. Advances in Neural Information Processing Systems 28 (NIPS 2015).

The AG's news topic classification dataset is constructed by choosing 4 largest classes from the original corpus. Each class contains 30,000 training samples and 1,900 testing samples. The total number of training samples is 120,000 and testing 7,600.

The file classes.txt contains a list of classes corresponding to each label.

The files train.csv and test.csv contain all the training samples as comma-sparated values. There are 3 columns in them, corresponding to class index (1 to 4), title and description. The title and description are escaped using double quotes ("), and any internal double quote is escaped by 2 double quotes (""). New lines are escaped by a backslash followed with an "n" character, that is "\n".

## Results

| Model Name    | Details       | Training Accuracy | Validation Accuracy | Best Epochs | Training Validation Plot |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| RNN with embedding as training process  | In this we create embedding as part of training process only  | 87.4 | 80.0 | 16 | [Plot1](https://github.com/Ashwinikumar1/NLP-DL/blob/master/Survey%20of%20Sequential%20Models%20on%20Text%20Classification/train_test_accuracy_plot/RNN_embedding_trained_in_model.png)|
| RNN with Glove Embedding (300)  | In this we only train RNN but use embedding as non trainable | 85.38 | 85.38 | 10 | [Plot2](https://github.com/Ashwinikumar1/NLP-DL/blob/master/Survey%20of%20Sequential%20Models%20on%20Text%20Classification/train_test_accuracy_plot/RNN_glove_embedding.png)|
| LSTM with Glove Embedding (300)  | In this we only train LSTM instead of RNN| 86.8 | 86.7 | 17 | [Plot3](https://github.com/Ashwinikumar1/NLP-DL/blob/master/Survey%20of%20Sequential%20Models%20on%20Text%20Classification/train_test_accuracy_plot/LSTM_glove_Embedding.png)|
| LSTM with Glove Embedding (300) and drop out in embedding layer  | In this we add 20% dropout before embeddings are inputetd to LSTMS| 87.40 | 87.9 | 40 | [Plot4](https://github.com/Ashwinikumar1/NLP-DL/blob/master/Survey%20of%20Sequential%20Models%20on%20Text%20Classification/train_test_accuracy_plot/LSTM_glove_embedding_dropouts.png)|


Next Steps :

1. Try GRU with dropout and without dropout
2. Try Biderectional LSTMs with and without dropout

