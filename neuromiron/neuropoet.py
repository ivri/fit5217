import re
import operator
import random
import random
import json
import sys
from collections import defaultdict

import nltk
import tensorflow
from pyphonetics import Soundex
from transliterate import translit
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import RMSprop
#from tensorflow.keras.utils.data_utils import get_file
import numpy as np

STYLE_FILENAME = "./oxxxy.txt"
HAWKING_CONTENT_FILENAME = "./hawking.txt"
UK_CONTENT_FILENAME = "./uk.txt"
WORD_RE = re.compile(r'[а-яА-ЯёЁa-zA-Z]+')

sys.path.append("/data/projects/punim0322/FIT5217/stuff/research/neuromiron/poet-ex-machina/")


import includes.accentsandsyllables as accentsandsyllables
from includes.utils import Utils
from includes.rhymesandritms import RhymesAndRitms
accents = accentsandsyllables.AccentsAndSyllables()
soundex = Soundex()


def load(*args):
    for varname in args:
        with open("{}.txt".format(varname), "r") as f:
            yield json.loads(f.read())
            
def save(**kwargs):
    for varname, data in kwargs.items():
        with open("{}.txt".format(varname), "w") as f:
            f.write(json.dumps(data))

def pos_accent_soundex(word):
    if len(word) > 1:
        sdx = soundex.phonetics(translit(word, "ru", reversed=True))
    else:
        sdx = translit(word, "ru", reversed=True)
        
    num_syllables, accent_syllable = accents.getAccentsAndSyllablesWord(word)
    rel_accent = accent_syllable / num_syllables
    if rel_accent <= 0.4:
        accent = 0
    elif rel_accent <= 0.67:
        accent = 1
    else:
        accent = 2
        
    _, pos = nltk.pos_tag([word], lang="rus")[0]
    real_pos = pos
    if pos in ("NONLEX"):
        return word
    return "{}{}{}".format(pos[0], accent, sdx)

print(pos_accent_soundex("неваляшка"))


def pos_soundex(word):
    if word == "/":
        return "/"
    try:
        if len(word) > 1:
            sdx = soundex.phonetics(translit(word, "ru", reversed=True))
        else:
            sdx = translit(word, "ru", reversed=True)
    except:
        return ""
        
    _, pos = nltk.pos_tag([word], lang="rus")[0]
    if pos in ("NONLEX"):
        return "X{}".format(sdx)
    return "{}{}".format(pos[0], sdx)

print(pos_soundex("неваляшка"))


def syllables_soundex(word):
    if word == "/":
        return "/"
    
    try:
        if len(word) > 1:
            sdx = soundex.phonetics(translit(word, "ru", reversed=True))
        else:
            sdx = translit(word, "ru", reversed=True)
    except:
        return ""
        
    syllables = len(Utils.getWordSyllables(word))
    return "{}{}".format(syllables, sdx)

print(syllables_soundex("неваляшка"))

def pos_syllables_soundex(word):
    if word == "/":
        return "/"
    
    try:
        if len(word) > 1:
            sdx = soundex.phonetics(translit(word, "ru", reversed=True))
        else:
            sdx = translit(word, "ru", reversed=True)
    except:
        return ""
    
    syllables = len(Utils.getWordSyllables(word))
    
    _, pos = nltk.pos_tag([word], lang="rus")[0]
    if pos in ("NONLEX"):
        return "X{}{}".format(syllables, sdx)
        
    return "{}{}{}".format(pos[0], syllables, sdx)

print(pos_syllables_soundex("неваляшка"))

def accent_soundex(word):
    if word == "/":
        return "/"
    
    if len(word) > 1:
        sdx = soundex.phonetics(translit(word, "ru", reversed=True))
    else:
        sdx = translit(word, "ru", reversed=True)
        
    num_syllables, accent_syllable = accents.getAccentsAndSyllablesWord(word)
    rel_accent = accent_syllable / num_syllables
    if rel_accent <= 0.4:
        accent = 0
    elif rel_accent <= 0.67:
        accent = 1
    else:
        accent = 2
        
    return "{}{}".format(accent, sdx)

print(accent_soundex("неваляшка"))

def levenshtein_distance(word1, word2):
    if len(word1) < len(word2):
        return levenshtein_distance(word2, word1)

    if len(word2) == 0:
        return len(word1)

    previous_row = list(range(len(word2) + 1))

    for i, char1 in enumerate(word1):
        current_row = [i + 1]

        for j, char2 in enumerate(word2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (char1 != char2)

            current_row.append(min(insertions, deletions, substitutions))

        previous_row = current_row
    return previous_row[-1]

def is_rhyme(word, rhymed_word):
    word_num_syllables, word_accent_syllable = accents.getAccentsAndSyllablesWord(word)
    rhymed_word_num_syllables, rhymed_word_accent_syllable = accents.getAccentsAndSyllablesWord(rhymed_word)
    
    rhymed_word_accented_syll_offset = rhymed_word_num_syllables - rhymed_word_accent_syllable
    rhymed_end_1 = RhymesAndRitms.getRhymedEnd(rhymed_word, rhymed_word_accented_syll_offset)
    word_accented_syll_offset = word_num_syllables - word_accent_syllable
    rhymed_end_2 = RhymesAndRitms.getRhymedEnd(word, word_accented_syll_offset)
    
    if not rhymed_end_1 or not rhymed_end_2:
        return False
    
#     if len(rhymed_end_1) != len(rhymed_end_2):
#         return False
    
    j = len(rhymed_end_2) - 1
    for i in range(len(rhymed_end_1) - 1, -1, -1):
        if j < 0:
            return True

        c1 = rhymed_end_1[i]
        c2 = rhymed_end_2[j]
#         print("c1=", c1, "c2=", c2)
        consonant = Utils.getConsonant(c2)
        if c1 != c2 and c1 != consonant and i > 1:
            return False

        j = j - 1

    return True

print(is_rhyme("блянина", "ссанина"))
print(is_rhyme("лиан", "виан"))
print(is_rhyme("говняшка", "неваляшка"))
print(is_rhyme("пидор", "залупа"))




style_corpus = []
with open(STYLE_FILENAME) as f:
    for line in f:
        cleared_line = " ".join(WORD_RE.findall(line))
        if cleared_line:
            style_corpus += [w for w in cleared_line.lower().split(" ") if w and not w.isnumeric()]
            style_corpus.append("/")

print(style_corpus[:100])

style_corpus_soundex = {}
style_corpus_soundex_list = []
for w in style_corpus:
    sdx = pos_soundex(w)
    style_corpus_soundex_list.append(sdx)
    style_corpus_soundex[sdx] = w

print(style_corpus_soundex_list[:100])

style_corpus_syllables = {}
style_corpus_syllables_list = []
for w in style_corpus:
    sdx = syllables_soundex(w)
    style_corpus_syllables_list.append(sdx)
    style_corpus_syllables[sdx] = w
    
print(style_corpus_syllables_list[:100])

style_corpus_pos_syllables = {}
style_corpus_pos_syllables_list = []
for w in style_corpus:
    sdx = pos_syllables_soundex(w)
    style_corpus_pos_syllables_list.append(sdx)
    style_corpus_pos_syllables[sdx] = w
    
print(style_corpus_pos_syllables_list[:100])

save(
    style_corpus=style_corpus, 
    style_corpus_soundex=style_corpus_soundex,
    style_corpus_soundex_list=style_corpus_soundex_list,
    style_corpus_syllables=style_corpus_syllables,
    style_corpus_syllables_list=style_corpus_syllables_list,
    style_corpus_pos_syllables=style_corpus_pos_syllables,
    style_corpus_pos_syllables_list=style_corpus_pos_syllables_list
)

    
    
def get_rhyme_end(word):
    word_num_syllables, word_accent_syllable = accents.getAccentsAndSyllablesWord(word)
    word_accented_syll_offset = word_num_syllables - word_accent_syllable
    return RhymesAndRitms.getRhymedEnd(word, word_accented_syll_offset)


hawking_content_corpus = []
with open(HAWKING_CONTENT_FILENAME) as f:
    for line in f:
        cleared_line = " ".join(WORD_RE.findall(line))
        if cleared_line:
            hawking_content_corpus += [w for w in cleared_line.lower().split(" ") if w and len(w) > 1 and not w.isnumeric()]

hawking_content_corpus = list(set(hawking_content_corpus))
print(hawking_content_corpus[:100])
hawking_content_corpus_soundex = {pos_soundex(w): w for w in hawking_content_corpus if w}
print(list(hawking_content_corpus_soundex.keys())[:100])
# hawking_content_corpus_accents = {accent_soundex(w): w for w in hawking_content_corpus if w}
# print(list(hawking_content_corpus_accents.keys())[:100])
hawking_content_corpus_syllables = {syllables_soundex(w): w for w in hawking_content_corpus if w}
print(list(hawking_content_corpus_syllables.keys())[:100])
hawking_content_corpus_pos_syllables = {pos_syllables_soundex(w): w for w in hawking_content_corpus if w}
print(list(hawking_content_corpus_pos_syllables.keys())[:100])

save(
    hawking_content_corpus=hawking_content_corpus, 
    hawking_content_corpus_soundex=hawking_content_corpus_soundex,
    #hawking_content_corpus_accents=hawking_content_corpus_accents,
    hawking_content_corpus_syllables=hawking_content_corpus_syllables,
    hawking_content_corpus_pos_syllables=hawking_content_corpus_pos_syllables
)


text = " ".join(style_corpus_pos_syllables_list)
chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 25
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))
print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.005)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
for iteration in range(1, 50):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y,
              batch_size=128,
              epochs=2)

    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = '' 
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(300):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()


model.save('oxxxy_pos_syllables_model.h6')


def generate_rnn(count):
    generated = ""
    start_index = random.randint(0, len(text) - maxlen - 1)
    sentence = text[start_index: start_index + maxlen]
    generated += sentence

    for i in range(count):
        x = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char
    return generated


def check_rhyme_end_with_word(rhymed_end_1, word):
    word_num_syllables, word_accent_syllable = accents.getAccentsAndSyllablesWord(word)
    word_accented_syll_offset = word_num_syllables - word_accent_syllable
    rhymed_end_2 = RhymesAndRitms.getRhymedEnd(word, word_accented_syll_offset)
    
    if not rhymed_end_1 or not rhymed_end_2:
        return False
    
    j = len(rhymed_end_2) - 1
    for i in range(len(rhymed_end_1) - 1, -1, -1):
        if j < 0:
            return True

        c1 = rhymed_end_1[i]
        c2 = rhymed_end_2[j]
        consonant = Utils.getConsonant(c2)
#         print("i=", i, "c1=", c1, "c2=", c2, "cons=", consonant)
        if c1 != c2 and c1 != consonant and i > 1:
            return False

        j = j - 1

    return True

rhymed_end = get_rhyme_end("калина")
print(rhymed_end)
print(check_rhyme_end_with_word(rhymed_end, "малина"))
print(check_rhyme_end_with_word(rhymed_end, "свинина"))
print(check_rhyme_end_with_word(rhymed_end, "сено"))


content = hawking_content_corpus_soundex
cache = set()
lines = []
rhymed_ends = defaultdict(list)
iteration = 0
while iteration < 100:
    iteration += 1
    lines = generate_rnn(10000).split("/")
    print("generated", len(lines))
    for line in lines:
        terms = [t for t in line.split(" ") if t]
        if not terms:
            continue
        
        generated_line = []
        for term in terms[:-1]:
            guessed_words = {}
            for idx, word in content.items():
                lev_dist = levenshtein_distance(idx, term)
                if len(word) > 1 and lev_dist <= 1:
                    guessed_words[word] = lev_dist
            
            if not guessed_words:
                for idx, word in style_corpus_soundex.items():
                    lev_dist = levenshtein_distance(idx, term)
                    if lev_dist <= 1:
                        guessed_words[word] = lev_dist
                        
            if guessed_words:
                generated_line.append(sorted(guessed_words, key=guessed_words.get)[0])
        
        last_words = []
        last_term = terms[-1]
        for idx, word in content.items():
            lev_dist = levenshtein_distance(idx, last_term)
            if len(word) > 1 and lev_dist <= 1:
                last_words.append(word)
                
        generated_line_str = " ".join(generated_line)
        for last_word in last_words:
            if last_word in rhymed_ends:
                continue
            
            rhymed_end = get_rhyme_end(last_word)
            for word in rhymed_ends.keys():
                if check_rhyme_end_with_word(rhymed_end, word):
                    rhymed_ends[word].append("{} {}".format(generated_line_str, last_word))
                    break
            else:
                rhymed_ends[last_word].append("{} {}".format(generated_line_str, last_word))
            
    print("Iteration {} completed".format(iteration))

for values in rhymed_ends.values():
    if len(values) > 3:
        print("-" * 30)
        for value in values:
            print(value)

