import numpy as np
import matplotlib.pyplot as plt

# example signal
signal = np.array([0.92,-0.88,0.91,-0.89,0.93,-0.87])

# function to add noise
def add_noise(signal, snr_db):

    signal_power = np.mean(signal**2)

    snr_linear = 10**(snr_db/10)

    noise_power = signal_power / snr_linear

    noise = np.sqrt(noise_power) * np.random.randn(len(signal))

    noisy_signal = signal + noise

    return noisy_signal


snr_levels = [20,10,5,0,-5]

for snr in snr_levels:

    noisy_signal = add_noise(signal,snr)

    print("SNR:",snr,"dB")
    print("Noisy Signal:",noisy_signal)
    print()