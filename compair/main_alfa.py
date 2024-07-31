from pydub import AudioSegment 
from mutagen.mp3 import MP3
from audio2vec import Audio2Vec
import numpy as np
from scipy.spatial import distance

processor = Audio2Vec()


one = MP3("compair/one.mp3")
pa = MP3("compair/pa_pa_pa.mp3")
song = AudioSegment.from_file("compair/pa_pa_pa.mp3", format="mp3")
one_length = one.info.length

oneVector = processor.audio2VectorProcessor('compair/one.mp3')
oneVector = np.squeeze(oneVector.toarray())


cut = one_length
line = 0
print(type(cut))

while pa.info.length >= cut:
    second = song[line:cut]
    second.export("compair/second.mp3",  format="mp3") 
    
    secondVector = processor.audio2VectorProcessor('compair/second.mp3')
    secondVector = np.squeeze(secondVector.toarray())

    print("Сравнениие слов Один и отрезка: ", distance.cosine(oneVector, secondVector))
    print("Начало обрезания: ", line)
    print("Конец обрезания: ", cut)
    print('-------------------------------------------------')

    line = line + one_length
    cut = cut + one_length

