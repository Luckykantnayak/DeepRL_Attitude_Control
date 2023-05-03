import matplotlib.pyplot as plt
import numpy as np
# Read data from .txt file
filename = 'Spacecraft_DDQN_UNIF_4.txt'  # Update with your actual file name
data = []
with open(filename, 'r') as f:
    for line in f:
        data.append(float(line.strip()))

data = np.array(data)
# Calculate running average
window_size = 500
running_avg = np.zeros(len(data))
running_std = np.zeros(len(data))
for i in range(len(data)):
    if i < window_size:
        running_avg[i] = np.mean(data[:i+1])
        running_std[i] = np.std(data[:i+1])
    else:
        running_avg[i] = np.mean(data[i-window_size+1:i+1])
        running_std[i] = np.std(data[i-window_size+1:i+1])

# Plot running average
# Plot data and running average/half std
fig, ax = plt.subplots()
ax.plot(running_avg, label='Running average',color='orange',linewidth=2)
ax.fill_between(np.arange(len(data)), running_avg-running_std/2, running_avg+running_std/2, alpha=0.5, label='Half std',color='blue' ,linewidth=2)
ax.legend(fontsize=18,loc='upper left')
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_ylabel('Returns', fontsize=24)
ax.set_xlabel('Episodes', fontsize=24)
ax.set_title("TD3 Controller Training", fontsize=30)

plt.show()
