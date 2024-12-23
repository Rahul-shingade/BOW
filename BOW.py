from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import re  # re will use for regular experssion
import nltk

x = '''Neural Networks are based on how the human brain works:
Neurons are sending messages to each other. While the neurons 
are trying to solve a problem (over and over again), it is strengthening 
the connections that lead to success and diminishing the connections that lead 
to failure.The First Layer:
The yellow perceptrons are making simple decisions based 
on the input. Each single decision is sent to the perceptrons 
in the next layer.The Second Layer:
The blue perceptrons are making decisions by weighing 
the results from the first layer. This layer make more complex decisions
 at a more abstract level than the first layer.
 Deep Neural Networks are made up of several hidden layers
 of neural networks that perform complex operations on massive amounts of data.
Each successive layer uses the preceding layer as input.
For instance, optical reading uses low layers to identify edges,
 and higher layers to identify letters.In the Deep Neural Network Model,
 input data (yellow)
 are processed against a hidden layer (blue) and modified against 
 more hidden layers (green) to produce the final output (red).
The First Layer:
The yellow perceptrons are making simple decisions 
based on the input. Each single decision is sent to the perceptrons in 
the next layer.
The Second Layer:
The blue perceptrons are making decisions by weighing the result
s from the first layer. This layer make more complex decisions at a 
more abstract level than the first layer.
The Third Layer:
Even more complex decisions are made by the green perceptrons.
Deep Learning (DL)
Deep Learning is a subset of Machine Learning.
Deep Learning is responsible for the AI boom of the last years.
Deep learning is an advanced type of ML that handles complex tasks
 like image recognition.
'''

# first step is lower sentence
# text cleaning is very important ,clean means remove the stop words
# lematization is for meaningful words and steming is for root node


ps = PorterStemmer()
wordnet = WordNetLemmatizer()
sent = nltk.sent_tokenize(x)
corpus = []

# create the one empty list bcz after clean the data store in the list

for i in range(len(sent)):
   review = re.sub('[^a-zA-Z]',' ',sent[i])
   review = review.lower()
   review = review.split()
   # review =[ps.stem(word) for word in review if not word in set(stopwords)]
   review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
   review = ''.join(review)
   corpus.append(review)
   
# create the bag of words model 

# also we called as document amtrix
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
out = cv.fit_transform(corpus).toarray()
