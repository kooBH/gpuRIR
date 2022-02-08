
import numpy as np
import librosa
import soundfile as sf


x1,_ = librosa.load("left2right.wav",sr=16000,mono=False)
x2,_ = librosa.load("back2front.wav",sr=16000,mono=False)

print(x1.shape)
print(x2.shape)

len_x1 = x1.shape[1]
len_x2 = x2.shape[1]

print(len_x1)
print(len_x2)

x2 = x2 * 0.5

if len_x1 > len_x2 :
    y = x1 + np.pad(x2,((0,0),(0,len_x1-len_x2)))
else :
    y = x2 + np.pad(x1,((0,0),(0,len_x2-len_x1)))

y= y/np.max(np.abs(y))

print(y.shape)

sf.write(file="output.wav",data=y.T,samplerate=16000)