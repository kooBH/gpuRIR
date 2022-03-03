from scipy.io import wavfile
import numpy as np
import os




def dB_scailing(denom,numer,dB):
    fs, d = wavfile.read("input/"+f1)
    fs, n1 = wavfile.read("input/"+f1)
    fs, n2 = wavfile.read("input/"+f1)
    fs, n3 = wavfile.read("input/"+f1)

    ## Normalization

    x1 = x1/np.max(np.abs(x1))
    x2 = x1/np.max(np.abs(x2))
    x3 = x1/np.max(np.abs(x3))
    x4 = x1/np.max(np.abs(x4))

    ## Calculate Energy

    ## Adjust SNR

    ## Apply SNR

    # Save Files
    os.makedirs("output".exist_ok=True)

    wavfile.write("output/"+f1, fs, y1)
    wavfile.write("output/"+f1, fs, y2)
    wavfile.write("output/"+f1, fs, y3)
    wavfile.write("output/"+f1, fs, y4)


if __name__ == "__main__":