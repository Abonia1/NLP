
import nltk

from gensim.models import Word2Vec
from nltk.corpus import stopwords

import re

paragraph = """There are three reasons why I prefer jogging to other sports. One reason is that jogging is a cheap sport. I can practise it anywhere at any time with no need for a ball or any other equipment. Another reason why I prefer jogging is that it is friendly to my heart. I don’t have to exhaust myself or do excessive efforts while jogging. Finally, I prefer this sport because it is safe. It isn’t as risky as other sports like gymnastics, racing or horseback riding. For all these reasons, I consider jogging the best sport of all.
There are three reasons why Canada is one of the best countries in the world. First, Canada has an excellent health care system. All Canadians have access to medical services at a reasonable price. Second, Canada has a high standard of education. Students are taught by well-trained teachers and are encouraged to continue studying at university. Finally, Canada's cities are clean and efficiently managed. Canadian cities have many parks and lots of space for people to live. As a result, Canada is a desirable place to live.
Aspirin can be a fatal poison. People are used to taking aspirin whenever they feel pain. It is true that aspirin is an efficacious pain-killer for example in headache cases. However, aspirin is like any other medicine can be dangerously harmful. Any unregulated use of it may result into the damage to the lining of the stomach, prolonged bleeding time, nausea, vomiting, ulcers, liver damage, and hepatitis. It is scientifically proven that excessive use of aspirin turns it into a toxin. Its toxic effects are Kidney Damage, severe metabolic derangements, respiratory and central nervous system effects, strokes, fatal haemorrhages of the brain, intestines & lungs and eventually death. Thus, the careful and regulated use of aspirin is most advisable so as not to turn into a deadly poison."""
               



# Preprocessing the data
text = re.sub(r'\[[0-9]*\]',' ',paragraph)
text = re.sub(r'\s+',' ',text)
text = text.lower()
text = re.sub(r'\d',' ',text)
text = re.sub(r'\s+',' ',text)

# Preparing the dataset
sentences = nltk.sent_tokenize(text)

sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]
    
    
# Training the Word2Vec model
model = Word2Vec(sentences, min_count=1)


words = model.wv.vocab

# Finding Word Vectors
vector = model.wv['war']

# Most similar words
similar = model.wv.most_similar('vikram')