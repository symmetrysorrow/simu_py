import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
import sys

def GetFFT(data,rate,sample):
    g_f=np.fft.fft(data)/fs
    frequencies=np.fft.fftfreq(int(sample), d=1/rate)
    g_f_shifted = np.fft.fftshift(g_f)
    frequencies_shifted = np.fft.fftshift(frequencies)
    return g_f_shifted,frequencies_shifted
def GetIFFT(data,rate):
    g_ifft=np.fft.ifft(data)*fs
    return g_ifft

fs = 1e6  # サンプリング周波数 (Hz)
N=1e6
t = np.linspace(0, 1, int(fs), endpoint=False)  # 時間軸
M_t=np.loadtxt("f:/hata/1332_adaptive/1332keV_17/Pulse/CH0/CH0_1.dat")
N_t=np.loadtxt("f:/hata/1332_adaptive/noise_time_domain.dat")
padded_data = np.pad(M_t, (10000, 0), mode='constant')  # 先頭に0を追加
M_t = padded_data[:-10000]  # 最後のn個を削除
D_t=M_t+N_t
# フーリエ変換
M_f,frequencies = GetFFT(M_t,fs,N)
N_f,frequencies=GetFFT(N_t,fs,N)
D_f,frequencies=GetFFT(D_t,fs,N)
ratio_squared = np.abs(M_f / N_f)**2

# 時間領域でのエネルギー計算
energy_time = np.sum(D_t**2) * (t[1] - t[0])  # 数値積分

# 周波数領域でのエネルギー計算
power_spectrum_M = np.abs(D_f)**2  # パワースペクトル
energy_freq = simpson(power_spectrum_M, x=frequencies)  # 数値積分

# 結果の表示
print(f"時間領域でのエネルギー: {energy_time}")
print(f"周波数領域でのエネルギー: {energy_freq}")

ratio_squared = np.abs(M_f / N_f)**2

C=simpson(ratio_squared,x=frequencies)


# 結果の出力
print(f"∫|M/N|^2 df = {C}")

# 結果のプロット
plt.figure(figsize=(12, 6))

plt.plot(t,D_t,label="pulse+noise")
plt.plot(t,N_t,label="Noise")
plt.plot(t, M_t, label="pulse")
plt.title('Time Domain Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.show()

plt.plot(frequencies,np.abs(D_f),label="pulse+noise")
plt.plot(frequencies, np.abs(M_f),label="Pulse")
plt.plot(frequencies,np.abs(N_f),label="Noise")
plt.title('Frequency Domain Signal')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

Template=GetIFFT(np.conj(M_f)/((np.abs(N_f))**2),N)

plt.plot(t,np.abs(Template))
plt.xlabel("Time[s]")
plt.show()