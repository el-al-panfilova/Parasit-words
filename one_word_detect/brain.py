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

#Сделать one.mp3 вектором
processor = Audio2Vec()
oneDataDF = processor.audio2Vec2DfProcessor('one_word_detect/one.mp3')
oneDataList = processor.audio2ListProcessor('one_word_detect/one.mp3')
oneVector = processor.audio2VectorProcessor('one_word_detect/one.mp3')
x_1, sr = librosa.load('one_word_detect/one.mp3')
#Сделать many.mp3 вектором
manyDataDF = processor.audio2Vec2DfProcessor('one_word_detect/many.mp3')
manyDataList = processor.audio2ListProcessor('one_word_detect/many.mp3')
manyVector = processor.audio2VectorProcessor('one_word_detect/many.mp3')

#
transform = au.AddGaussianSNR(
    min_snr_db= 0.5,
    max_snr_db= 40.0,
    p=1.0
)

print(oneVector.shape)
augmented_sound = transform(x_1 ,sample_rate=1600)

#sound = pydub.AudioSegment.from_wav(augmented_sound)
wav_io = io.BytesIO()
scipy.io.wavfile.write(wav_io, 16000, augmented_sound)
wav_io.seek(0)
sound = pydub.AudioSegment.from_wav(wav_io)
print(type(sound))
sound.export("one_word_detect/apple.mp3", format="mp3")

one_remade_vec = processor.audio2VectorProcessor('one_word_detect/apple.mp3')


print(oneVector.shape, one_remade_vec.shape, type(oneVector.toarray()))
oneVector = np.squeeze(oneVector.toarray())
one_remade_vec = np.squeeze(one_remade_vec.toarray())
print(oneVector.shape, one_remade_vec.shape, type(oneVector))

print(distance.cosine(oneVector,one_remade_vec))

