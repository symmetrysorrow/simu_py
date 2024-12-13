import csv
import numpy as np
import json
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import natsort
import glob
import re
import tqdm

def ReadOutput(FilePath,key):
    df=pd.read_csv(FilePath)
    data = df[key].to_numpy()
    return data

def remove_outliers(data, percentile=0.1):
    lower_bound = np.percentile(data, percentile)
    upper_bound = np.percentile(data, 100 - percentile)
    return data[(data >= lower_bound) & (data <= upper_bound)]

def ReadPulse(Data_path,pulse,path):
    with open(f"{Data_path}/input.json") as f:
        input=json.load(f)
    try:
        peak = np.max(pulse)
        peak_index = np.argmax(pulse)
        peak_av = np.mean(pulse[peak_index - 10 : peak_index +90])

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

        ST_height = np.mean(pulse[input["SettlingTime"] - 10 : input["SettlingTime"] +90])

        return [peak_av,peak_index,rise,ST_height]
    except:
        print(path)

def MakeOutput(Data_path,target):
    with open(f"{Data_path}/input.json") as f:
        input = json.load(f)
    for posi in tqdm.tqdm(input["position"]):
        for ch in[0,1]:
            results=[]
            pulse_numbers=[]
            pulse_pathes = natsort.natsorted(glob.glob(f'{Data_path}/{input["E"]}keV_{posi}/{target}/CH{ch}/CH{ch}_*.dat'))

            for path in pulse_pathes:
                pattern = fr'CH{ch}_(\d+).dat'
                match = re.search(pattern, path)
                pulse_numbers.append(match.group(1))

                pulse=np.loadtxt(path)
                results.append(ReadPulse(Data_path,pulse,path))

            columns=["height","peak_index","rise","ST_Height"]
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

def optimal_bin_count(data):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1  # 四分位範囲
    bin_width = 2 * iqr / (len(data) ** (1/3))  # ビン幅
    bin_count = int(np.ceil((np.max(data) - np.min(data)) / bin_width))  # ビン数
    return max(bin_count, 1)  # ビン数が1未満にならないようにする

def MakeHistgram(data,posi):
    bin_num = optimal_bin_count(data)
    hist, bin_edges = np.histogram(data, bins=bin_num, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # ビンの中心を計算
    initial_guess = [np.max(hist), np.mean(data), np.std(data)]

    plt.hist(data, bins=bin_num, density=False, alpha=0.6, label=f"abs-{posi}")

    # ガウスフィッティング
    popt, pcov = curve_fit(gaussian, bin_centers, hist, p0=initial_guess, maxfev=10000000)
    amp_fit, mean_fit, stddev_fit = popt
    fwhm = 2 * stddev_fit * np.sqrt(2 * np.log(2))

      # ヒストグラム
    x_fit = np.linspace(bin_edges[0], bin_edges[-1], 1000)  # フィッティング用のx
    plt.plot(x_fit, gaussian(x_fit, *popt),color="red",alpha=0.5)  # フィッティング曲線
    return fwhm,fwhm/mean_fit

def Resos(Data_path,target,show):
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

    for CH0,CH1,posi in zip(CH0_files,CH1_files,para["position"]):
        CH0_heights=ReadOutput(CH0,"height")
        CH1_heights=ReadOutput(CH1,"height")
        
        ratios=CH0_heights/CH1_heights

        positions=find_nearest_values(ratios,fit_para)
        #positions=remove_outliers(positions)
        
        fwhm,reso=MakeHistgram(positions,posi)
        fwhms.append(fwhm)

    fwhms=np.array(fwhms)
    np.savetxt(f"{Data_path}/fwhms_{target}.txt",fwhms)

    if show:
        plt.xlabel("Position",fontsize=15)
        plt.ylabel("Count",fontsize=15)
        plt.tight_layout()
        #plt.legend()
        plt.savefig(f"{Data_path}/position_histgram_{target}.png")
        plt.show()
        plt.plot(para["position"],fwhms)
        plt.show()

    ene_reso_sum=[]

    for CH0,CH1,posi in zip(CH0_files,CH1_files,para["position"]):
        CH0_heights=ReadOutput(CH0,"height")
        CH1_heights=ReadOutput(CH1,"height")
        heights=CH0_heights+CH1_heights
        heights=remove_outliers(heights)
        fwhm,reso=MakeHistgram(heights,posi)
        ene_reso_sum.append(reso*para["E"])

    if show:
        plt.xlabel("Current[A]",fontsize=15)
        plt.ylabel("Count",fontsize=15)
        plt.tight_layout()
        plt.legend()
        plt.savefig(f"{Data_path}/energy_sum_histgram_{target}.png")
        plt.show()
        plt.plot(para["position"],ene_reso_sum)
        plt.show()

    ene_reso_max=[]

    for CH0,CH1,posi in zip(CH0_files,CH1_files,para["position"]):
        CH0_heights=ReadOutput(CH0,"height")
        CH1_heights=ReadOutput(CH1,"height")
        heights=np.maximum(CH0_heights,CH1_heights)
        fwhm,reso=MakeHistgram(heights,posi)
        ene_reso_max.append(reso*para["E"])
    if show:
        plt.xlabel("Current[A]",fontsize=15)
        plt.ylabel("Count",fontsize=15)
        plt.tight_layout()
        #plt.legend()
        plt.savefig(f"{Data_path}/energy_max_histgram_{target}.png")
        plt.show()
        plt.plot(para["position"],ene_reso_max)
        plt.show()

    ene_reso_min=[]

    for CH0,CH1,posi in zip(CH0_files,CH1_files,para["position"]):
        CH0_heights=ReadOutput(CH0,"height")
        CH1_heights=ReadOutput(CH1,"height")
        heights=np.minimum(CH0_heights,CH1_heights)
        fwhm,reso=MakeHistgram(heights,posi)
        ene_reso_min.append(reso*para["E"])
    if show:
        plt.xlabel("Current[A]",fontsize=15)
        plt.ylabel("Count",fontsize=15)
        plt.tight_layout()
        #plt.legend()
        plt.savefig(f"{Data_path}/energy_min_histgram_{target}.png")
        plt.show()
        plt.plot(para["position"],ene_reso_min)
        plt.show()

    ene_reso_st=[]

    for CH0,CH1,posi in zip(CH0_files,CH1_files,para["position"]):
        CH0_heights=ReadOutput(CH0,"ST_Height")
        CH1_heights=ReadOutput(CH1,"ST_Height")
        heights=CH0_heights+CH1_heights
        fwhm,reso=MakeHistgram(heights,posi)
        ene_reso_st.append(reso*para["E"])
    if show:
        plt.xlabel("Current[A]",fontsize=15)
        plt.ylabel("Count",fontsize=15)
        plt.tight_layout()
        #plt.legend()
        plt.savefig(f"{Data_path}/energy_ch_histgram_{target}.png")
        plt.show()
        plt.plot(para["position"],ene_reso_st)
        plt.show()

    data={
        "Sum":ene_reso_sum,
        "Max": ene_reso_max,
        "Min":ene_reso_min,
        "ST":ene_reso_st
    }

    index_label=np.array(para["position"])
    index_label=(index_label-1/2)*para["length"]/para["n_abs"]

    df=pd.DataFrame(data,index=index_label)
    df.to_csv(f"{Data_path}/ene_resos_{target}.csv")


show=False
Data_path="f:/hata/662_142_136"
#MakeOutput(Data_path,"Pulse_noise")
#MakeOutput(Data_path,"Pulse_ms")
#MakeOutput(Data_path,"Pulse_ms_noise")
#Resos(Data_path,"Pulse_noise",show)
Resos(Data_path,"Pulse_ms",show)
Resos(Data_path,"Pulse_ms_noise",show)