# In a Vector Space of La Mancha, whose Position I Do Wish to Recall... 🍇

<img width="874" alt="espacio" src="https://github.com/mariagabv/lasolana-embeddings/assets/107461889/fa1f8fc7-e135-4ee9-b54e-b2fb54f135ba">


This repository presents a **comprehensive evaluation of word embeddings** in Spansih with paper and code. We trained word vectors over [La Solana](https://es.wikipedia.org/wiki/La_Solana) using the library `fasttext`. You can visualize them following this link to [Tensorflow Embeddings Projector](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/mariagabv/lasolana-embeddings/main/lasolana-embeddings/tensorflow-projector/lasolana-embeddings_config.json). Try searching for 'la_casota', 'ofrecimiento' or 'moje'. Do you have any idea what these words mean in La Solana?

Summary and links for lasolana-embeddings:

| Corpus                     | Size    | Algorithm | Vectors | Vec-Dim |
|----------------------------|---------|-----------|----------|---------|
| [Collected corpus](https://github.com/mariagabv/lasolana-embeddings/blob/main/preprocessing/preprocessing_EDA.ipynb) | 4.1M    | FastText  | 29682| 150     |

## La Solana FastText embeddings
Links to the embeddings (#dimensions=150, #vectors=29,682):

- [Vector format](https://github.com/mariagabv/lasolana-embeddings/blob/main/lasolana-embeddings/lasolana_embeddings.vec) (.vec) (34.5 MB)
- [Binary format](https://github.com/mariagabv/lasolana-embeddings/blob/main/lasolana-embeddings/lasolana_embeddings.bin) (.bin) (34.5 MB)
- [TSV format](https://github.com/mariagabv/lasolana-embeddings/blob/main/lasolana-embeddings/lasolana_embeddings.tsv) (.tsv) (49.5 MB)

## Corpus
All digitized corpus about La Solana that we have access to:
- [Gaceta de La Solana](https://www.lasolana.es/ayuntamiento/radio-horizonte/la-gaceta) until december 2023
- [Noticias de La Solana](https://www.lasolana.es/laciudad/noticias)
- [Julián Simón's blog](https://joaquincostalasolanalegadobustillo.blogspot.com/)
- [García Gallego, José María. El legado Bustillo de La Solana: su fundación, visicitudes, intervención de Costa, asesinato del cura Torrijos, situación actual.(1935)
](https://ceclmdigital.uclm.es/results.vm?q=id:0000330120&lang=es&view=libros)
  
Corpus Size: over 4 million words
Preprocessing: Explained in [training_eval](https://github.com/mariagabv/lasolana-embeddings/blob/main/training_eval/train_eval.ipynb) notebook.

## Algorithm
[Implementation](https://github.com/mariagabv/lasolana-embeddings/blob/main/training_eval/train_eval.ipynb): FastText with Skipgram and no sub-words

### Parameters
- min subword-ngram = 0
- max subword-ngram = 0
- neg = 5
- ws = 5
- epochs = 5
- dim = 150
- all other parameters set as default
