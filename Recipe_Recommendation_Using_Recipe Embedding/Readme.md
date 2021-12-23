### Recipe Recommendation - Using NLP (In absence of any cutomer data)

The objective of this work is to use free text of recipes from food-com to create a NLP engine which can identify similar recipes based on Semantic similarity of recipe text. Once we have encodes recipes we can apply varipus algorithm and network methods to create a recommendation engine

The USP of this work is that we are not using any Customer preference or transation data to create this recommendation engine

### About Data : 
This dataset consists of 180K+ recipes and 700K+ recipe reviews covering 18 years of user interactions and uploads on Food.com (formerly GeniusKitchen). used in the following paper:

Generating Personalized Recipes from Historical User Preferences

Bodhisattwa Prasad Majumder*, Shuyang Li*, Jianmo Ni, Julian McAuley

EMNLP, 2019

https://www.aclweb.org/anthology/D19-1613/

Statistics About Data :
1. Number of Distinct Recipes : ~231K
2. Number of Distinct Users : ~119K

After looking into data we observed following :

1.  Raw Interactions data shows that more than 95% of ratings are > 4 which clearly indicates some Bias.Further analysis shows that most of ratings are from person posting the recipes. 

2.  For our POC, we have filtered out only those recipes which have more than 2 reviewers. This leaves us with only 94k recipes. (This is just POC thats why otherwise it makes more sense to keep al recipes which can help us discover New Unknown Recipes)

### Data Preparation :
1.Sumamrise the Raw interaction data at recipes level and count unique number of users who have reviewed it
2.The Raw Recipes data we have has recice text as list of steps, conbine by recipes id to form a single string
3.Join tables from step 1 & step at recipe level(recipe level data) and keep only those recipes where number of reviewer are greater than 2
4. Now the data is ready we will start creating Recipe Embedding

### Embedding Creation:

From my previous experience, in Quora Questions Pairs etc word level embedding does not explain complete sentences properly. Also, i have seen from my previous experience though BERT cls embedding alsd does not give state of the art results.

So in our current work we will use pretrained Siamese Bert Model, which has been fine tuned for differentiate between similar sentences semantically. It builds over BERT model by taking average of all the word vectors and finetuning them to fit the training task. The model architecture is defined below:

![Sbert-Model](https://github.com/Ashwinikumar1/NLP-DL/blob/master/Recipe_Recommendation_Using_Recipe%20Embedding/Outputs/Model_Sbert.JPG) 

As we did not have any taggings we have used pretrained models. But if we have good amount of tag data we can use 


### Algorithm 1 : Given a recipe return top 4 similar recipes
In this function we make use of cosine similarity between input recipe  and all other recipes vector and return top 4 recipes with highest cosine similarity. It is interesting to note that with this we get similar recipes without any transaction data

Also, the similarity varies at various level. if you look at examples below

Example 1 : You get prodcts similar based on ingredients but also based on type i.e Desserts
Example 2 : It returns recipes which are similar because they follows same steps
Example 3 : It returns you burgers with differenr preprations and ingredients
Example 4 : It return products which have similar main ingredients i.e. Potato but different prep strategies
Example 5 : For Taco, you get all mexican recipes because maybe they have similar preparation strategy

The results are as follows :
![Recommendations](https://github.com/Ashwinikumar1/NLP-DL/blob/master/Recipe_Recommendation_Using_Recipe%20Embedding/Outputs/Product_Recommendations.PNG)
 
The problem with this algorithm is we cannot applying commutative properties (If A is similar to B and B is similar to C) to this is quite expensive operation. We will make use of Network to solve this issue

### Algorithm 2 : Create a complete network using cosine similarity between Recipes as edges and Recipes as Nodes. Currently we have restriced edges to top 5 similar recipes.The network help us mine certain important relationships.

Some examples are as follows :If someone likes 3 Hour Old Fashioned Pot Roast We may recommend him Apple Bacon Thanks Giving Dressing or Flaky Suasage Foldovers even when they don't have direct relationship between them

![recommendation1](https://github.com/Ashwinikumar1/NLP-DL/blob/master/Recipe_Recommendation_Using_Recipe%20Embedding/Outputs/Network_Recommendation1.JPG)

Example 2 : If some likes Chicken Wings or Artichoke chickens you amy recommend him 5 star chicken sun dried tomato afred or angry chicken though they don't have direct link 

![recommendation2](https://github.com/Ashwinikumar1/NLP-DL/blob/master/Recipe_Recommendation_Using_Recipe%20Embedding/Outputs/Network_Recommedation2.JPG)

You can explore various variations of graphs exported in pdf in outputs folder

Though the results makes a lot of sense, if i had more time i could have tried follwoing to further improve

1. Use other information i.e Tags Data and ingredient data which calculating similarity of products

2. Give user a functionaliy to select recipes based on ingredient, preparation or both

3. Instead of using pretrained embedding we van train our own embedding if we had similarity data

Refrences :

1. https://arxiv.org/abs/1908.10084 : I have used only this paper otherwise complete project was just my idea



 

