import numpy as np

# Example received signal
signal = np.array([0.92,-0.88,0.91,-0.89,0.93,-0.87])

# ---------------------------
# BPSK Demodulation
# ---------------------------
def demodulate_bpsk(signal):
    bits = []
    for s in signal:
        if s >= 0:
            bits.append(1)
        else:
            bits.append(0)
    return bits


# ---------------------------
# QPSK Demodulation
# ---------------------------
def demodulate_qpsk(signal):
    bits = []
    for s in signal:
        if s >= 0:
            bits.extend([1,1])
        else:
            bits.extend([0,0])
    return bits


# ---------------------------
# Predicted modulation type
# ---------------------------
predicted_modulation = "BPSK"   # CNN prediction result

# ---------------------------
# Demodulation
# ---------------------------
if predicted_modulation == "BPSK":
    recovered_bits = demodulate_bpsk(signal)

elif predicted_modulation == "QPSK":
    recovered_bits = demodulate_qpsk(signal)

else:
    recovered_bits = []

# ---------------------------
# Output
# ---------------------------
print("Predicted Modulation:", predicted_modulation)
print("Recovered Digital Data:", recovered_bits)

# ---------------------------
# Save recovered bits to file
# ---------------------------
with open("recovered_data.txt","w") as f:
    for bit in recovered_bits:
        f.write(str(bit))