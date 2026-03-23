import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

data = pd.read_csv("signals.csv")

for i,row in data.iterrows():

    signal = np.array(list(map(float,row["signal"].split())))

    # increase signal length
    signal = np.tile(signal,200)

    label = row["label"]

    folder = "spectrograms/"+label

    if not os.path.exists(folder):
        os.makedirs(folder)

    f,t,Sxx = spectrogram(signal,nperseg=64)

    plt.figure(figsize=(2,2))
    plt.pcolormesh(t,f,10*np.log10(Sxx+1e-10), shading='gouraud')
    plt.axis("off")

    plt.savefig(folder+"/img"+str(i)+".png",
                bbox_inches='tight',
                pad_inches=0)

    plt.close()

print("Spectrogram images created")