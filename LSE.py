import json
import numpy as np
from scipy.optimize import minimize
import pulse_model
import general as gn
import matplotlib.pyplot as plt

# --- 設定 ---
sim_path = "h:/hata2025/200_180"
exp_path="G:/TSURUTA/20230616_post/room1-ch2-3_180mK_570uA_100kHz_g10"
eta = 101 #(μA/V)
amp = 100
dt = 1.0  # サンプリング周期（秒単位にする場合は適切な値を入力）

# --- 特徴量抽出関数 ---
def extract_features(pulse, dt=1.0):
    y = pulse - np.mean(pulse[:500])
    h_max = np.max(y)
    if h_max <= 0: return 0.0, 0.0, 1e-9 # 0ではなく微小値を返す

    idx_max = np.argmax(y)
    
    # --- 減衰時定数 (tau) の計算を「割合」ベースに変更 ---
    try:
        # ピーク以降のデータのみ抽出
        decay_part = y[idx_max:]
        
        # 高さが 80% まで落ちた点から 20% まで落ちた点の間でフィッティング
        # これにより「急なオレンジ」も「緩いグレー」も同じルールで測れる
        start_idx = np.where(decay_part <= h_max * 0.8)[0][0]
        end_idx   = np.where(decay_part <= h_max * 0.2)[0][0]
        
        fit_y = decay_part[start_idx:end_idx]
        fit_t = np.arange(len(fit_y)) * dt
        
        if len(fit_y) > 5:
            # 対数線形フィッティング
            slope, _ = np.polyfit(fit_t, np.log(fit_y), 1)
            tau = -1.0 / slope if slope < 0 else 1e-9
        else:
            tau = 1e-9 # データ不足時は極小値
    except:
        tau = 1e-9
        
    # 高さ、立ち上がり、時定数
    return h_max, (np.argmax(y[:idx_max+1] >= h_max*0.9) - np.argmax(y[:idx_max+1] >= h_max*0.1)) * dt / len(pulse), tau

# --- シミュレーション実行関数 ---
def MakePulse(para,Presample):   
    ch0, ch1 = pulse_model.model(para)
    # 必要に応じて position 分の pulses を返す
    pulses=[]
    for pulse in ch0:
        zeros = np.zeros(int(Presample))
        pulse=np.concatenate([zeros, pulse])[:int(para["samples"])]
        pulses.append(pulse)
    return pulses

# --- メイン処理 ---

# 1. 設定の読み込み
with open(f"{sim_path}/input.json", "r") as f:
    para = json.load(f)

# 2. 実験データのロードと特徴量抽出

selected_keys=gn.LoadTxt(f"{exp_path}/SelectedKeys_fromScatter.txt")
para_exp=gn.LoadJson(f"{exp_path}/PulseConfig.json")
Presample=int(para_exp["Readout"]["PreSample"])
dt=para_exp["Readout"]["Sample"]/para_exp["Readout"]["Rate"]

peak_values = []
keys_list = []

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

# 抽出
pulse_high = gn.LoadBin(f"{exp_path}/CH0_pulse/rawdata/CH0_{int(key_high)}.dat")*eta/amp/1e6
pulse_high-= np.mean(pulse_high[0:Presample])
pulse_low = gn.LoadBin(f"{exp_path}/CH0_pulse/rawdata/CH0_{int(key_low)}.dat")*eta/amp/1e6
pulse_low -= np.mean(pulse_low[0:Presample])
exp_h_high, exp_tr_high, exp_tau = extract_features(pulse_high, dt)
exp_h_low, exp_tr_low, _ = extract_features(pulse_low, dt)

print(f"Highest Pulse: Height = {exp_h_high}, Rise Time = {exp_tr_high}, Tau = {exp_tau}")
print(f"Lowest Pulse : Height = {exp_h_low}, Rise Time = {exp_tr_low}")

time=gn.GetTime(para_exp["Readout"]["Rate"], para_exp["Readout"]["Sample"])

pulses = MakePulse(para, Presample)

pulse_sim_high = pulses[0]
pulse_sim_low = pulses[-1]
print(f" Initial Simulation High Pulse: Height = {extract_features(pulse_sim_high, dt)[0]}, Rise Time = {extract_features(pulse_sim_high, dt)[1]}, Tau = {extract_features(pulse_sim_high, dt)[2]}")
print(f" Initial Simulation Low Pulse : Height = {extract_features(pulse_sim_low, dt)[0]}, Rise Time = {extract_features(pulse_sim_low, dt)[1]}, Tau = {extract_features(pulse_sim_low, dt)[2]}")

plt.plot(time, pulse_high, label="Experimental High Pulse", color="gray", linestyle="--")
plt.plot(time, pulse_low, label="Experimental Low Pulse", color="gray", linestyle="--")
plt.plot(time, pulse_sim_high, label="Simulated High Pulse")
plt.plot(time, pulse_sim_low, label="Simulated Low Pulse")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()


# 3. 評価関数 (ここが悩みの解決策)
def err_func(params):

    # パラメータ更新
    para["C_abs"] = params[0]
    para["C_tes"] = params[1]
    para["G_abs-abs"] = params[2]
    para["G_abs-tes"] = params[3]
    para["G_tes-bath"] = params[4]
    
    # 負の値を防ぐ制約（Nelder-Mead用）
    if any(p <= 0 for p in params):
        return 1e18

    # シミュレーション実行
    try:
        pulses = MakePulse(para, Presample)
        sim_h_low, sim_tr_low, _ = extract_features(pulses[-1], dt)
        sim_h_high, sim_tr_high, sim_tau = extract_features(pulses[0], dt)
    except:
        return 1e18

    # --- 5項目の相対誤差計算 ---
    # 分母に微小値を足してZeroDivisionを防ぐ
    e_h_high = ((sim_h_high - exp_h_high) / (exp_h_high))**2
    e_tr_high = ((sim_tr_high - exp_tr_high) / (exp_tr_high))**2
    e_h_low = ((sim_h_low - exp_h_low) / (exp_h_low))**2
    e_tr_low = ((sim_tr_low - exp_tr_low) / (exp_tr_low))**2
    e_tau = ((sim_tau - exp_tau) / (exp_tau))**2

    # 重み付け: スパイク（高さと立ち上がり）を重視
    # [h_high, tr_high, h_low, tr_low, tau]
    weights = np.array([10.0, 5.0, 10.0, 0.0, 10.0])
    errors = np.array([e_h_high, e_tr_high, e_h_low, e_tr_low, e_tau])
    
    total_error = np.sum(weights * errors)
    return total_error

# 4. 最適化の実行
initial_params = [7.9e-10, 7.9e-12, 1.5e-07, 8.2e-09, 1.68e-08]

result = minimize(err_func, initial_params, method="Nelder-Mead", 
                  options={'maxiter': 500, 'disp': True, 'xatol': 1e-8, 'fatol': 1e-8})

print("Optimization Result:")
print(result.x)

# 最終結果を保存
para["C_abs"], para["C_tes"], para["G_abs-abs"], para["G_abs-tes"], para["G_tes-bath"] = result.x
with open(f"{sim_path}/final_params.json", "w") as f:
    json.dump(para, f,indent=4,)

pulses = MakePulse(para, Presample)

print(f" High Pulse Simulated: Height = {extract_features(pulses[0], dt)[0]}, Rise Time = {extract_features(pulses[-1], dt)[1]}, Tau = {extract_features(pulses[-1], dt)[2]}")
print(f" Low Pulse Simulated: Height = {extract_features(pulses[-1], dt)[0]}, Rise Time = {extract_features(pulses[0], dt)[1]}, Tau = {extract_features(pulses[0], dt)[2]}")

plt.plot(time, pulse_high, label="Experimental High Pulse", color="gray", linestyle="--")
plt.plot(time, pulse_low, label="Experimental Low Pulse", color="gray", linestyle="--")
plt.plot(time, pulses[0], label="Simulated High Pulse")
plt.plot(time, pulses[-1], label="Simulated Low Pulse")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()