import matplotlib.pyplot as plt

snr_levels = [20,15,10,5,0,-5]

# example accuracy values
accuracy = [98,96,92,85,70,55]

plt.plot(snr_levels,accuracy,marker='o')

plt.xlabel("SNR (dB)")
plt.ylabel("Classification Accuracy (%)")
plt.title("Accuracy vs SNR")

plt.grid(True)

plt.show()