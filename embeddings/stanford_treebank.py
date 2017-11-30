from __future__ import absolute_import
import csv
__PATH__='/home/ubik/data/stanford_sentiment'
dictiobary_file = __PATH__ + '/dictionary.txt'
sentiment_labels_file = __PATH__ + '/sentiment_labels.txt'


def load_stanford():
    id_to_phrase = {}
    id_to_sentiment = {}
    with open(dictiobary_file) as f:
        reader = csv.reader(f, delimiter='|')
        next(reader)
        for row in reader:
            text = row[0]
            id = row[1]
            id_to_phrase[id] = text

    with open(sentiment_labels_file) as f:
        reader = csv.reader(f, delimiter='|')
        next(reader)
        for row in reader:
            id = row[0]
            sentiment = float(row[1])
            id_to_sentiment[id] = sentiment

    return id_to_phrase, id_to_sentiment

id_to_phrase, id_to_sentiment = load_stanford()