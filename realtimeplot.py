"""
License: MIT
Created by Kurene@wizard-notes.com
"""
import time
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

plt.style.use('dark_background')


# Parameters (audio)
sr         = 44100
n_ch       = 2
filepath   = "your_audiofile.wav"
n_fft      = 4096
hop_length = int(n_fft / 4 * 3) # 0 <= hop_length < n_fft
overlap    = n_fft - hop_length
n_plot_tf  = 80
n_freqs    = n_fft//2 + 1
f_max_idx  = 480 # 1 < f_max_idx < n_freqs
window     = np.hamming(n_fft)
amp        = np.zeros((n_plot_tf, f_max_idx))

# Parameters (plot, video)
fps = 1.0
fig, ax = plt.subplots()
image = ax.imshow(amp.T, aspect="auto")
ax.set_xlabel(f"Time frame")
ax.set_ylabel(f"Frequency")
fig.colorbar(image)
vmax, vmin = 1.0, 0.0
#min_fps = sr / hop_length

pretime = time.time()
for index, block in enumerate(sf.blocks(filepath, blocksize=n_fft, overlap=overlap)):
    if block.shape[0] != n_fft:
        continue
    
    x = np.mean(block, axis=1) # to monoral
    amp[-1] = np.sqrt(np.abs(np.fft.rfft(window * x)))[0:f_max_idx]
    if vmax < np.max(amp[-1]):
        vmax = np.max(amp[-1])
    image.set_clim(vmin, vmax)
    image.set_data(amp.T[::-1])
    
    #plt.title(f"fps: {fps:0.1f} Hz\n(min. fps requirement: {min_fps:0.1f} Hz)")
    plt.title(f"fps: {fps:0.1f} Hz")
    plt.pause(0.001)

    amp[0:-1] = amp[1::]

    curtime = time.time()
    time_diff = curtime - pretime
    fps = 1.0/(time_diff + 1e-16)
    #print(f"{index}:\t{time_diff:0.3f} sec")
    pretime = curtime
    
plt.close()
