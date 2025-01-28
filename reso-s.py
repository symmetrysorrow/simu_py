import csv
import numpy as np
import json
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import UnivariateSpline
import natsort
import glob
import re
import tqdm
import getpara as gp

def ReadOutput(FilePath,key):
    df=pd.read_csv(FilePath)
    data = df[key].to_numpy()
    return data

def remove_outliers(data, percentile=0.1):
    lower_bound = np.percentile(data, percentile)
    upper_bound = np.percentile(data, 100 - percentile)
    return data[(data >= lower_bound) & (data <= upper_bound)]

def ReadPulse(Data_path,pulse,path,target):
    with open(f"{Data_path}/input.json") as f:
        para=json.load(f)
    if target != "Pulse_ms":
        pulse=gp.BesselFilter(pulse,para["rate"],para["cutoff"])
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

        rise = (rise_90 - rise_10) / para["rate"]

        ST_height = np.mean(pulse[para["SettlingTime"] - 10 : para["SettlingTime"] +90])

        return [peak_av,peak_index,rise,ST_height]
    except:
        print(path)

def MakeOutput(Data_path,target):
    with open(f"{Data_path}/input.json") as f:
        para = json.load(f)

    if target != "Pulse_ms":
        print("BesselFilter")
    for posi in tqdm.tqdm(para["position"]):
        for ch in[0,1]:
            results=[]
            pulse_numbers=[]
            pulse_pathes = natsort.natsorted(glob.glob(f'{Data_path}/{para["E"]}keV_{posi}/{target}/CH{ch}/CH{ch}_*.dat'))

            for path in pulse_pathes:
                pattern = fr'CH{ch}_(\d+).dat'
                match = re.search(pattern, path)
                pulse_numbers.append(match.group(1))

                pulse=np.loadtxt(path)
                results.append(ReadPulse(Data_path,pulse,path,target))

            columns=["height","peak_index","rise","ST_Height"]
            df = pd.DataFrame(results,columns=columns,index=pulse_numbers)
            df.to_csv(f"{Data_path}/{para["E"]}keV_{posi}/{target}/output_TES{ch}.csv")

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

def MakeHistgram(data,posi,HistColor=None):
    bin_num = optimal_bin_count(data)
    #bin_num=30
    hist, bin_edges = np.histogram(data, bins=bin_num, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # ビンの中心を計算
    initial_guess = [np.max(hist), np.mean(data), np.std(data)]
    if HistColor is not None:
        plt.hist(data, bins=bin_num, density=False, label=f"abs-{posi}",color=HistColor)
    else:
        plt.hist(data, bins=bin_num, density=False, label=f"abs-{posi}")
    # ガウスフィッティング
    popt, pcov = curve_fit(gaussian, bin_centers, hist, p0=initial_guess, maxfev=10000000)
    amp_fit, mean_fit, stddev_fit = popt
    fwhm = 2 * stddev_fit * np.sqrt(2 * np.log(2))

    #ヒストグラム
    x_fit = np.linspace(bin_edges[0], bin_edges[-1], 1000)  # フィッティング用のx
    plt.plot(x_fit, gaussian(x_fit, *popt),color="red",alpha=0.5)  # フィッティング曲線

    #plt.axvline(x=mean_fit)
    #plt.show()
    return fwhm,fwhm/mean_fit

def generate_symmetric_colors(n):
    # 色相を0から1の範囲で分布させる（広い色相範囲で）
    half_n = n // 2  # n//2 で配列の半分を計算
    hues = np.linspace(0.1, 0.9, half_n)  # 色相範囲を広げる（赤から紫まで）
    hues = np.concatenate([hues, hues[::-1]])  # 左右対称にする

    # 奇数の場合、中央の色を赤色（hues[half_n]）に設定
    if n % 2 != 0:
        hues = np.concatenate([hues[:half_n], [0.0], hues[half_n:]])  # 中央を赤に

    # 彩度と明度を一定に保つ（高彩度と高明度で明るく）
    saturation = 1.0
    value = 1.0
    
    # HSVからRGBに変換
    colors = [mcolors.hsv_to_rgb((h, saturation, value)) for h in hues]
    
    return np.array(colors)

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

    ratio_base = fit_para[:, 1]
    posi_base = fit_para[:, 0]

    sorted_indices = np.argsort(ratio_base)
    ratio_base = ratio_base[sorted_indices]
    posi_base = posi_base[sorted_indices]

    spline = UnivariateSpline(ratio_base, posi_base, k=3, s=0)

    for CH0,CH1,posi in zip(CH0_files,CH1_files,para["position"]):
        CH0_heights=ReadOutput(CH0,"height")
        CH1_heights=ReadOutput(CH1,"height")
        
        ratios=CH0_heights/CH1_heights

        positions=find_nearest_values(ratios,fit_para)

        positions=spline(ratios)
        #positions=remove_outliers(positions)
        
        fwhm,reso=MakeHistgram(positions,posi)
        fwhms.append(fwhm)

    fwhms=np.array(fwhms)
    np.savetxt(f"{Data_path}/fwhms_{target}.txt",fwhms)

    
    
    plt.xlabel("Position",fontsize=15)
    plt.ylabel("Count",fontsize=15)
    plt.title(f"{para["E"]}keV")
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"{Data_path}/position_histgram_{target}_{para["E"]}.png")
    if show:
        plt.show()
        plt.plot(para["position"],fwhms)
        plt.show()
    else:
        plt.clf()

    coeffs = np.polyfit(ratio_base, posi_base, 15)
    poly_func = np.poly1d(coeffs)  # 15次多項式関数を作成

    fwhms=[]

    for CH0,CH1,posi in zip(CH0_files,CH1_files,para["position"]):
        CH0_heights=ReadOutput(CH0,"height")
        CH1_heights=ReadOutput(CH1,"height")
        
        ratios=CH0_heights/CH1_heights

        positions = poly_func(ratios)
        
        fwhm,reso=MakeHistgram(positions,posi)
        fwhms.append(fwhm)

    fwhms=np.array(fwhms)
    np.savetxt(f"{Data_path}/fwhms_{target}_15.txt",fwhms)

    if show:
        plt.cla()
        plt.plot(para["position"],fwhms)
        plt.show()


show=False
Data_path="F:/hata/1332_142_136_300split"
#MakeOutput(Data_path,"Pulse_noise")
#MakeOutput(Data_path,"Pulse_ms")
#MakeOutput(Data_path,"Pulse_ms_noise")
#MakeOutput(Data_path,"pulse_noise_ms_test")
#Resos(Data_path,"Pulse_noise",show)
Resos(Data_path,"Pulse_ms",show)
#Resos(Data_path,"Pulse_ms_noise",show)
#Resos(Data_path,"pulse_noise_ms_test",show)