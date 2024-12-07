import csv
import numpy as np
import json
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import natsort
import glob
import re

def ReadOutput(FilePath,key):
    df=pd.read_csv(FilePath)
    data = df[key].to_numpy()
    return data

def ReadPulse(Data_path,pulse,path):
    with open(f"{Data_path}/input.json") as f:
        input=json.load(f)
    try:
        peak = np.max(pulse)
        peak_index = np.argmax(pulse)
        peak_av = np.mean(pulse[peak_index - 3 : peak_index +7])

        for i in reversed(range(0, peak_index)):
            if pulse[i] <= peak * 0.9:
                rise_90 = i
                break

        try:
            rise_90+=0
        except:
            rise_90=0
        for j in reversed(range(0, rise_90)):
            if pulse[j] <= peak * 0.1:
                rise_10 = j
                break

        try:
            rise_10+=0
        except:
            rise_10=0

        rise = (rise_90 - rise_10) / input["rate"]

        CheckPoint_height = np.mean(pulse[input["CheckPoint"] - 3 : input["CheckPoint"] +7])

        return [peak_av,peak_index,rise,CheckPoint_height]
    except:
        print(path)

def MakeOutput(Data_path,target):
    print("normal")
    with open(f"{Data_path}/input.json") as f:
        input = json.load(f)
    for posi in input["Position"]:
        for ch in[0,1]:
            results=[]
            pulse_numbers=[]
            pulse_pathes = natsort.natsorted(glob.glob(f'{Data_path}/{input["E"]}keV_{posi}/{target}/CH{ch}/CH{ch}_*.dat'))

            for path in pulse_pathes:
                pattern = fr'CH{ch}_(\d+).dat'
                match = re.search(pattern, path)
                pulse_numbers.append(match.group(1))

                pulse=np.loadtxt(path)
                results.append(ReadPulse(Data_path,pulse))

            columns=["height","peak_index","rise","CheckPointHeight"]
            df = pd.DataFrame(results,columns=columns,index=pulse_numbers)
            df.to_csv(f"{Data_path}/{input["E"]}keV_{posi}/{target}/output_TES{ch}.csv")

def gaussian(x, amp, mean, stddev):
    return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

def find_nearest_values(ratios, Fit_para):
    # B の右側の列
    B_right = Fit_para[:, 1]
    # B の左側の列
    B_left = Fit_para[:, 0]

    # 結果を格納するリスト
    result = []

    for a in ratios:
        # Aの値に最も近いB_rightの値のインデックスを見つける
        idx = np.abs(B_right - a).argmin()
        # 対応するB_leftの値を結果に追加
        result.append(B_left[idx])

    return np.array(result)  # numpy配列として返す

def MakeHistgram(data,bin_num):
    hist, bin_edges = np.histogram(data, bins=bin_num, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # ビンの中心を計算
    initial_guess = [np.max(hist), np.mean(data), np.std(data)]

    # ガウスフィッティング
    popt, pcov = curve_fit(gaussian, bin_centers, hist, p0=initial_guess, maxfev=10000)
    amp_fit, mean_fit, stddev_fit = popt
    fwhm = 2 * stddev_fit * np.sqrt(2 * np.log(2))

    plt.hist(data, bins=bin_num, density=True, alpha=0.6, label="Histogram")  # ヒストグラム
    x_fit = np.linspace(bin_edges[0], bin_edges[-1], 1000)  # フィッティング用のx
    plt.plot(x_fit, gaussian(x_fit, *popt), label="Gaussian Fit", color="red")  # フィッティング曲線

    return fwhm

def Resos(Data_path,target,bin_num):
    fit_para_path=f"{Data_path}/ratios.txt"
    fit_para=np.loadtxt(fit_para_path, delimiter=',')

    with open(f"{Data_path}/input.json","r") as f:
        para=json.load(f)

    CH0_files=[]
    CH1_files=[]

    for posi in para["position"]:
        CH0_files.append(f"{Data_path}/{para['E']}keV_{posi}/{target}/output_TES0.csv")
        CH1_files.append(f"{Data_path}/{para['E']}keV_{posi}/{target}/output_TES1.csv")

    fwhms=[]
    plt.figure(figsize=(4,3))

    for CH0,CH1 in zip(CH0_files,CH1_files):
        CH0_heights=ReadOutput(CH0,"height")
        CH1_heights=ReadOutput(CH1,"height")
        
        ratios=CH0_heights/CH1_heights
        positions=find_nearest_values(ratios,fit_para)
        
        fwhm=MakeHistgram(positions,bin_num)
        fwhms.append(fwhm)
    
    plt.xlabel("Position",fontsize=15)
    plt.ylabel("Density",fontsize=15)
    plt.tight_layout()
    plt.savefig(f"{Data_path}/position_histgram_{target}.png")
    plt.show()
    fwhms=np.array(fwhms)
    np.savetxt(f"{Data_path}/fwhms_{target}.txt",fwhms)
    plt.plot(para["position"],fwhms)
    plt.ylim(0,max(fwhms)+0.1)
    plt.show()

    ene_reso_sum=[]

    for CH0,CH1 in zip(CH0_files,CH1_files):
        CH0_heights=ReadOutput(CH0,"height")
        CH1_heights=ReadOutput(CH1,"height")
        heights=CH0_heights+CH1_heights
        fwhm=MakeHistgram(heights,bin_num)
        ene_reso_sum.append(fwhm*para["E"])

    plt.xlabel("Position",fontsize=15)
    plt.ylabel("Density",fontsize=15)
    plt.tight_layout()
    plt.savefig(f"{Data_path}/energy_sum_histgram_{target}.png")
    plt.show()
    ene_reso_sum=np.array(ene_reso_sum)
    np.savetxt(f"{Data_path}/ene_reso_sum_{target}.txt",ene_reso_sum)
    plt.plot(para["position"],ene_reso_sum)
    plt.ylim(0,max(ene_reso_sum)+0.1)
    plt.show()

    ene_reso_max=[]

    for CH0,CH1 in zip(CH0_files,CH1_files):
        CH0_heights=ReadOutput(CH0,"height")
        CH1_heights=ReadOutput(CH1,"height")
        heights=np.maximum(CH0_heights,CH1_heights)
        fwhm=MakeHistgram(heights,bin_num)
        ene_reso_max.append(fwhm*para["E"])

    plt.xlabel("Position",fontsize=15)
    plt.ylabel("Density",fontsize=15)
    plt.tight_layout()
    plt.savefig(f"{Data_path}/energy_max_histgram_{target}.png")
    plt.show()
    ene_reso_max=np.array(ene_reso_max)
    np.savetxt(f"{Data_path}/ene_reso_max_{target}.txt",ene_reso_max)
    plt.plot(para["position"],ene_reso_max)
    plt.ylim(0,max(ene_reso_max)+0.1)
    plt.show()

    ene_reso_min=[]

    for CH0,CH1 in zip(CH0_files,CH1_files):
        CH0_heights=ReadOutput(CH0,"height")
        CH1_heights=ReadOutput(CH1,"height")
        heights=np.minimum(CH0_heights,CH1_heights)
        fwhm=MakeHistgram(heights,bin_num)
        ene_reso_min.append(fwhm*para["E"])

    plt.xlabel("Position",fontsize=15)
    plt.ylabel("Density",fontsize=15)
    plt.tight_layout()
    plt.savefig(f"{Data_path}/energy_min_histgram_{target}.png")
    plt.show()
    ene_reso_min=np.array(ene_reso_min)
    np.savetxt(f"{Data_path}/ene_reso_min_{target}.txt",ene_reso_min)
    plt.plot(para["position"],ene_reso_min)
    plt.ylim(0,max(ene_reso_min)+0.1)
    plt.show()

    ene_reso_ch=[]

    for CH0,CH1 in zip(CH0_files,CH1_files):
        CH0_heights=ReadOutput(CH0,"CheckPoint")
        CH1_heights=ReadOutput(CH1,"CheckPoint")
        heights=CH0_heights+CH1_heights
        fwhm=MakeHistgram(heights,bin_num)
        ene_reso_ch.append(fwhm*para["E"])

    plt.xlabel("Position",fontsize=15)
    plt.ylabel("Density",fontsize=15)
    plt.tight_layout()
    plt.savefig(f"{Data_path}/energy_ch_histgram_{target}.png")
    plt.show()
    ene_reso_ch=np.array(ene_reso_ch)
    np.savetxt(f"{Data_path}/ene_reso_min_{target}.txt",ene_reso_ch)
    plt.plot(para["position"],ene_reso_ch)
    plt.ylim(0,max(ene_reso_ch)+0.1)
    plt.show()


Data_path="d:/hata/phits/662_142_100"
MakeOutput(Data_path,"Pulse_noise")
MakeOutput(Data_path,"Pulse_ms")
MakeOutput(Data_path,"Pulse_ms_noise")
Resos(Data_path,"Pulse_noise",60)
Resos(Data_path,"Pulse_ms",60)
Resos(Data_path,"Pulse_ms_noise",60)