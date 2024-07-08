from __future__ import print_function
from audio2vec import Audio2Vec
import numpy as np
import matplotlib.pyplot as plt
import audiomentations as au
# from itertools import zip
from math import sqrt
from scipy.spatial import distance
import librosa
import librosa.display
from transformers import Wav2Vec2Model, AutoConfig, Wav2Vec2Processor
from datasets import load_dataset, load_metric
import pydub
import io
import scipy.io.wavfile

processor = Audio2Vec()

oneVector = processor.audio2VectorProcessor('one_word_detect/audio/one.mp3')
oneVector = np.squeeze(oneVector.toarray())
buscetVector = processor.audio2VectorProcessor('one_word_detect/audio/buscet.mp3')
buscetVector = np.squeeze(buscetVector.toarray())
houseVector = processor.audio2VectorProcessor('one_word_detect/audio/house.mp3')
houseVector = np.squeeze(houseVector.toarray())
punVector = processor.audio2VectorProcessor('one_word_detect/audio/pan.mp3')
punVector = np.squeeze(punVector.toarray())
scrudriverVector = processor.audio2VectorProcessor('one_word_detect/audio/scrudriver.mp3')
scrudriverVector  = np.squeeze(scrudriverVector .toarray())
one_another_Vector = processor.audio2VectorProcessor('one_word_detect/audio/apple.mp3')
one_another_Vector = np.squeeze(one_another_Vector.toarray())

print(distance.cosine(oneVector,scrudriverVector))
print(distance.cosine(oneVector,one_another_Vector))


