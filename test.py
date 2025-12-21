import json
import numpy as np
from scipy.optimize import minimize
import pulse_model
import general as gn
import matplotlib.pyplot as plt


sim_path = "h:/hata2025/200_180"
exp_path="G:/TSURUTA/20230616_post/room1-ch2-3_180mK_570uA_100kHz_g10"

selected_keys=gn.LoadTxt(f"{exp_path}/SelectedKeys_fromScatter.txt")
para_exp=gn.LoadJson(f"{exp_path}/PulseConfig.json")
Presample=int(para_exp["Readout"]["PreSample"])
dt=para_exp["Readout"]["Sample"]/para_exp["Readout"]["Rate"]

eta = 101 #(μA/V)
amp = 100
dt = 1.0  # サンプリング周期（秒単位にする場合は適切な値を入力）
peak_values = []
keys_list = []

def MakePulse(para,Presample):   
    ch0, ch1 = pulse_model.model(para)
    # 必要に応じて position 分の pulses を返す
    pulses=[]
    for pulse in ch0:
        zeros = np.zeros(int(Presample))
        pulse=np.concatenate([zeros, pulse])[:int(para["samples"])]
        pulses.append(pulse)
    return pulses

for key in selected_keys:
    pulse = gn.LoadBin(f"{exp_path}/CH0_pulse/rawdata/CH0_{int(key)}.dat")
    pulse = pulse.copy()
    pulse -= np.mean(pulse[0:Presample])
    pulse = gn.Bessel(pulse, para_exp["Readout"]["Rate"], para_exp["Analysis"]["CutoffFrequency"])
    
    # ピーク値を計算して保存
    peak_values.append(np.max(pulse))
    keys_list.append(key)

# numpy配列に変換して最大・最小のインデックスを探す
peak_values = np.array(peak_values)
max_idx = np.argmax(peak_values)
min_idx = np.argmin(peak_values)

# 最高のパルスと最低のパルスのkeyを取得
key_high = keys_list[max_idx]
key_low = keys_list[min_idx]

print(f"Highest Pulse: Key = {key_high}, Amplitude = {peak_values[max_idx]}")
print(f"Lowest Pulse : Key = {key_low}, Amplitude = {peak_values[min_idx]}")

# 抽出
pulse_high = gn.LoadBin(f"{exp_path}/CH0_pulse/rawdata/CH0_{int(key_high)}.dat")*eta/amp/1e6
pulse_high-= np.mean(pulse_high[0:Presample])
pulse_low = gn.LoadBin(f"{exp_path}/CH0_pulse/rawdata/CH0_{int(key_low)}.dat")*eta/amp/1e6
pulse_low -= np.mean(pulse_low[0:Presample])

para=gn.LoadJson(f"{sim_path}/final_params.json")

time=gn.GetTime(para_exp["Readout"]["Rate"], para_exp["Readout"]["Sample"])

plt.plot(time, pulse_high, label="Experimental High Pulse", color="gray", linestyle="--")
plt.plot(time, pulse_low, label="Experimental Low Pulse", color="gray", linestyle="--")
plt.plot(time, MakePulse(para, Presample)[-1], label="Simulated High Pulse")
plt.plot(time, MakePulse(para, Presample)[0], label="Simulated Low Pulse")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
#plt.legend()
plt.show()