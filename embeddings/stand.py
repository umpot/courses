# Set up spaCy
import io
import spacy
from collections import Counter

nlp = spacy.load('en')

# Test Data
ff = io.open('/home/ubik/data/Stephen King - The Stand - 1978.txt', mode="r", encoding="utf-8")
ff = ff.read()

doc = nlp(ff)

c = Counter([x.lemma_ for x in doc])
with io.open('/home/ubik/PycharmProjects/courses/embeddings/stand.txt', mode='w', encoding="utf-8") as f:
    m= c.most_common(len(c))
    for x, y in m:
        f.write(u'{} {}\n'.format(x,y))