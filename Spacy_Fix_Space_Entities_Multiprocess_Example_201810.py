'''
Demonstrate adding a rule-based component that forces some tokens to not
be entities, before the NER tagger is applied.
'''
import spacy
from spacy.attrs import ENT_IOB


def fix_space_tags(doc):
    # ENT_IOB: IOB code of named entity tag. "B" means the token begins an entity,
    #     "I" means it is inside an entity, "O" means it is outside an entity,
    #     and "" means no entity tag is set.
    ent_iobs = doc.to_array([ENT_IOB])
    for i, token in enumerate(doc):
        if token.is_space:
            # Sets 'O' tag (0 is None, so I is 1, O is 2)
            ent_iobs[i] = 2
    doc.from_array([ENT_IOB], ent_iobs.reshape((len(doc), 1)))
    return doc


nlp = spacy.load('en_core_web_sm')
text = u'''This is some crazy test where I dont need an Apple                Watch to make things bug'''
doc = nlp(text)
print('Before', doc.ents)
nlp.add_pipe(fix_space_tags, name='fix-ner', before='ner')
doc = nlp(text)
print('After', doc.ents)

# Expected print out:
# Before (Apple,                Watch)
# After (Apple,)




##############################################################################
##############################################################################
##############################################################################
"""
Example of multi-processing with Joblib.

Here, we're exporting part-of-speech-tagged, true-cased, (very roughly) sentence-separated text, with
each "sentence" on a newline, and spaces between tokens. Data is loaded from
the IMDB movie reviews dataset and will be loaded automatically via Thinc's
built-in dataset loader.

Compatible with: spaCy v2.0.0+
"""


from __future__ import print_function, unicode_literals
from toolz import partition_all
from pathlib import Path
from joblib import Parallel, delayed
import thinc.extra.datasets
import plac
import os
import spacy


@plac.annotations(
    output_dir=("Output directory", "positional", None, Path),
    model=("Model name (needs tagger)", "positional", None, str),
    n_jobs=("Number of workers", "option", "n", int),
    batch_size=("Batch-size for each process", "option", "b", int),
    limit=("Limit of entries from the dataset", "option", "l", int))
def main(output_dir, model='en_core_web_sm', n_jobs=4, batch_size=1000,
         limit=10000):
    nlp = spacy.load(model)  # load spaCy model
    print("Loaded model '%s'" % model)
    if not output_dir.exists():
        output_dir.mkdir()
    # load and pre-process the IMBD dataset
    print("Loading IMDB data...")
    data, _ = thinc.extra.datasets.imdb()
    texts, _ = zip(*data[-limit:])
    print("Processing texts...")
    partitions = partition_all(batch_size, texts)
    executor = Parallel(n_jobs=n_jobs)
    do = delayed(transform_texts)
    tasks = (do(nlp, i, batch, output_dir)
             for i, batch in enumerate(partitions))
    executor(tasks)


def transform_texts(nlp, batch_id, texts, output_dir):
    """
    This is the main precedure to perform the calculations for each batch
    """
    print(nlp.pipe_names)
    # ['tagger', 'parser', 'ner'], here tagger is to assign each token noun, verb, etc.
    #     parser is to assign the structure of senteces, such as subject or object of a verb
    #     ner: Named Entity Recognition (NER) labels sequences of words in a text which
    #         are the names of things, such as person and company names, or gene and protein names
    out_path = Path(output_dir) / ('%d.txt' % batch_id)
    if out_path.exists():  # return None in case same batch is called again
        return None
    print('Processing batch', batch_id)
    with out_path.open('w', encoding='utf8') as f:
        for doc in nlp.pipe(texts):
            f.write(' '.join(represent_word(w) for w in doc if not w.is_space))
            f.write('\n')
    print('Saved {} texts to {}.txt'.format(len(texts), batch_id))


def represent_word(word):
    text = word.text
    # True-case, i.e. try to normalize sentence-initial capitals.
    # Only do this if the lower-cased form is more probable.
    if text.istitle() and is_sent_begin(word) \
       and word.prob < word.doc.vocab[text.lower()].prob:
        text = text.lower()
    return text + '|' + word.tag_


def is_sent_begin(word):
    # Tell if the word is the beginning of the sentence
    if word.i == 0:
        return True
    elif word.i >= 2 and word.nbor(-1).text in ('.', '!', '?', '...'):
        return True
    else:
        return False


####################  MAIN PROCEDURE  ####################
model = "en_core_web_sm"
n_jobs=4
batch_size=1000
limit=10000
output_dir = Path(os.getcwd())

nlp = None  # clear the existing nlp model
nlp = spacy.load(model)  # load spaCy model
print("Loaded model '%s'" % model)
if not output_dir.exists(): output_dir.mkdir()
# load and pre-process the IMBD dataset
print("Loading IMDB data...")
data, _ = thinc.extra.datasets.imdb()  # data is a list of 25000 reviews
texts, _ = zip(*data[-limit:])  # texts is just last 10000 reviews from data
print("Processing texts...")
partitions = partition_all(batch_size, texts)  # partiton generator
executor = Parallel(n_jobs=n_jobs)  # prepare the parallelization
do = delayed(transform_texts)
tasks = (do(nlp, i, batch, output_dir)
         for i, batch in enumerate(partitions))
executor(tasks)


####### Decompose the individual taks
for doc in nlp.pipe(texts):
    x = ' '.join(represent_word(w) for w in doc if not w.is_space)
    break
print(x)
'''
This|DT is|VBZ very|RB nearly|RB a|DT perfect|JJ film|NN .|. The|DT ideas|NNS would|MD be|VB
repeated|VBN by|IN Mamet|NNP ,|, but|CC never|RB told|VBN so|RB succinctly|RB .|.
This|DT is|VBZ really|RB about|IN the|DT failure|NN of|IN trust|NN ,|, of|IN the|DT
human|JJ condition|NN .|. The|DT film|NN weaves|VBZ the|DT idea|NN that|IN we|PRP are|VBP
all|DT criminals|NNS ,|, no|DT one|NN is|VBZ innocent|JJ .|. Is|VBZ there|EX anyone|NN
alive|JJ today|NN who|WP has|VBZ n't|RB seen|VBN this|DT play|VB out|RP in|IN
our|PRP$ own|JJ society|NN ,|, every|DT single|JJ day|NN ?|. The|DT film|NN is|VBZ
very|RB much|RB structured|JJ like|IN a|DT Hitchcock|NNP thriller|NN .|.
Except|IN ,|, there|EX are|VBP no|DT more|RBR innocent|JJ characters|NNS .|.
The|DT world|NN is|VBZ now|RB completely|RB polluted|VBN ,|, ruined|VBN
and|CC everyone|NN is|VBZ participating|VBG in|IN the|DT con|NN .|. Could|MD
anything|NN be|VB more|RBR true|JJ ?|. Do|VBP n't|RB miss|VB the|DT
soundtrack|NN .|. It|PRP is|VBZ wonderful|JJ .|.
'''

# The non-parallel version
def _transform_texts(nlp, texts, output_dir):
    print(nlp.pipe_names)
    out_path = Path(output_dir) / ('zzz.txt')
    # write the text output to zzz.txt
    with out_path.open('w', encoding='utf8') as f:
        for doc in nlp.pipe(texts):
            f.write(' '.join(represent_word(w) for w in doc if not w.is_space))
            f.write('\n')

_transform_texts(nlp, texts, output_dir)

