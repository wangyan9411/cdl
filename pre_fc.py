import nltk
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer

buff = open('./event.json', 'r').read()

dic = eval(buff)
english_punctuations = [',', '.', ':', ';', '?', '--', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '-', '/', '+', '=', '...']
english_stopwords = stopwords.words('english')
st = LancasterStemmer()

documents = []

for event in dic:
    event = dic[event]
    desc = event['description'].lower()
    documents.append(desc)

print documents[0]
tokens = [nltk.word_tokenize(desc) for desc in documents]
print 'token '
print tokens[0]
tokens = [[word for word in t if not word in english_stopwords] for t in tokens]
print 'stop'
print tokens[0]
texts_filtered = [[word for word in t if not word in english_punctuations] for t in tokens]
print 'punctuation'
print texts_filtered[0]

texts_stemmed = [[st.stem(word) for word in text] for text in texts_filtered]
print 'stem'
print len(texts_stemmed[0])
print texts_stemmed[0]
all_stems = sum(texts_stemmed, [])
stems_once = set(stem for stem in set(all_stems) if all_stems.count(stem) == 1)
texts = [[stem for stem in text if stem not in stems_once] for text in texts_stemmed]
print 'once'
print texts[0]
print len(texts[0])
all_stems = sum(texts, [])
print len(all_stems)
all_stems = set(all_stems)
print len(all_stems)
dic = dict(zip(all_stems, range(len(all_stems))))

ls = []
for text in texts:
    all_words = text
    ls.append([str(dic[w]) + ':' + str(all_words.count(w)) for w in set(all_words)])

print ls[0]
ls = [str(len(l)) + ' ' + ' '.join(l) for l in ls]
context = '\n'.join(ls) 
open('text','w').write(context)

raw = '\n'.join([' '.join(text) for text in texts])
open('raw-text', 'w').write(raw)


