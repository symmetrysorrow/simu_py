import numpy as np
import matplotlib.pyplot as plt
import getpara as gp
from scipy.signal import butter, filtfilt

rate=1e6
samples=1e4
time = np.arange(samples) / rate

order = 4  # フィルタの次数
cutoff = 1e4  # カットオフ周波数 (Hz)
b, a = butter(order, cutoff / (rate / 2), btype='low')  # 正規化カットオフ周波数

pulses=[]

for i in range(0,99):
    pulses.append(np.loadtxt(f"F:/hata/1332_142_136_300split/1332keV_17/pulse_noise/CH0/CH0_{i}.dat"))
for i,pulse in enumerate(pulses):
    pulse=gp.BesselFilter(pulse,rate,1e4)
    #pulse=filtfilt(b, a, pulse)
    plt.plot(time,pulse,label=i)
plt.legend()
plt.show()