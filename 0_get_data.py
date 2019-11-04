import gluonnlp

import gluonnlp as nlp
train_dataset, test_dataset = [
    nlp.data.IMDB(root='data/imdb', segment=segment)
    for segment in ('train', 'test')
]
