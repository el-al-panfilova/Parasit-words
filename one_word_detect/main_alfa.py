from __future__ import print_function
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import librosa
import librosa.display

# First, let's load a first version of our audio recordings.
x_1, fs = librosa.load('one_word_detect/one.mp3')
plt.figure(figsize=(16, 4))
librosa.display.waveshow(x_1, sr=fs)
plt.title('One word Version $X_1$')
plt.tight_layout()


x_2, fs = librosa.load('one_word_detect/many.mp3')
plt.figure(figsize=(16, 4))
librosa.display.waveshow(x_2, sr=fs)
plt.title('Many word Version $X_1$')
plt.tight_layout()

# And a second version, slightly faster.
x_3, fs = librosa.load('one_word_detect/apple.mp3')
plt.figure(figsize=(16, 4))
librosa.display.waveshow(x_3, sr=fs)
plt.title('Apple words Version $X_2$')
plt.tight_layout()



plt.show()