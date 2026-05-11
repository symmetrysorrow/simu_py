import csv
import numpy as np
import json
import pandas as pd
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import UnivariateSpline
import natsort
import glob
import re
import tqdm
import getpara as gp


def save_readpulse_debug(Data_path, pulse_raw, pulse_filt, para, path, reason, peak_index=None, rise_10=None, rise_90=None):
    debug_dir = f"{Data_path}/debug_readpulse"
    os.makedirs(debug_dir, exist_ok=True)

    t = np.arange(len(pulse_raw)) / para["rate"]
    st_l = max(0, para["SettlingTime"] - 10)
    st_r = min(len(pulse_filt), para["SettlingTime"] + 90)

    plt.figure(figsize=(10, 5))
    plt.plot(t, pulse_raw, color="gray", alpha=0.35, label="raw")
    plt.plot(t, pulse_filt, color="navy", linewidth=1.2, label="filtered")

    if peak_index is not None and 0 <= peak_index < len(t):
        plt.axvline(t[peak_index], color="red", linestyle="--", label="peak")
    if rise_10 is not None and 0 <= rise_10 < len(t):
        plt.axvline(t[rise_10], color="green", linestyle="--", label="10%")
    if rise_90 is not None and 0 <= rise_90 < len(t):
        plt.axvline(t[rise_90], color="orange", linestyle="--", label="90%")

    if len(pulse_filt) > 0:
        peak = np.nanmax(pulse_filt)
        plt.axhline(peak * 0.1, color="green", alpha=0.25)
        plt.axhline(peak * 0.9, color="orange", alpha=0.25)

    plt.axvspan(st_l / para["rate"], st_r / para["rate"], color="gray", alpha=0.18, label="Settling window")
    plt.title(f"{path}\nreason: {reason}")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()

    safe_path = re.sub(r'[\\\\/:*?"<>|]', "_", str(path))
    plt.savefig(f"{debug_dir}/{safe_path}.png", dpi=200)
    plt.close()

def ReadOutput(FilePath,key):
    df=pd.read_csv(FilePath)
    data = df[key].to_numpy()
    return data

def remove_outliers(data, percentile=0.1):
    lower_bound = np.percentile(data, percentile)
    upper_bound = np.percentile(data, 100 - percentile)
    return data[(data >= lower_bound) & (data <= upper_bound)]

def ReadPulse(Data_path, pulse, target="Pulse_ms", path="", debug_plot=False):
    with open(f"{Data_path}/input.json") as f:
        para = json.load(f)

    if np.mean(pulse) <=0:
        pulse*= -1

    try:
        if target != "Pulse_ms":
            pulse_filt = gp.BesselFilter(pulse, para["rate"], para["cutoff"])
        else:
            pulse_filt = pulse

        peak = np.max(pulse_filt)
        peak_index = np.argmax(pulse_filt)

        # 明らかにおかしい波形は除外
        if peak_index < 10:
            raise ValueError("peak too early")

        # peak average
        l = max(0, peak_index - 10)
        r = min(len(pulse_filt), peak_index + 90)
        peak_av = np.mean(pulse_filt[l:r])

        # rise 90%
        rise_90 = None
        for i in reversed(range(0, peak_index)):
            if pulse_filt[i] <= peak * 0.9:
                rise_90 = i
                break
        if rise_90 is None:
            raise ValueError("rise_90 not found")

        # rise 10%
        rise_10 = None
        for j in reversed(range(0, rise_90)):
            if pulse_filt[j] <= peak * 0.1:
                rise_10 = j
                break
        if rise_10 is None:
            raise ValueError("rise_10 not found")

        rise = (rise_90 - rise_10) / para["rate"]

        # settling height
        st_l = para["SettlingTime"] - 10
        st_r = para["SettlingTime"] + 90
        ST_height = np.mean(pulse_filt[st_l:st_r])

        # =========================
        # debug plot
        # =========================
        if debug_plot:
            t = np.arange(len(pulse_filt)) / para["rate"]

            plt.figure(figsize=(8, 4))
            plt.plot(t, pulse_filt, label="pulse")

            plt.axvline(t[peak_index], color="r", linestyle="--", label="peak")
            plt.axvline(t[rise_10], color="g", linestyle="--", label="10%")
            plt.axvline(t[rise_90], color="orange", linestyle="--", label="90%")

            plt.axhline(peak * 0.1, color="g", alpha=0.3)
            plt.axhline(peak * 0.9, color="orange", alpha=0.3)

            plt.axvspan(
                st_l / para["rate"],
                st_r / para["rate"],
                color="gray",
                alpha=0.2,
                label="Settling window",
            )

            plt.xlabel("Time [s]")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.tight_layout()
            plt.show()

        return [peak_av, peak_index, rise, ST_height]

    except Exception as e:
        if debug_plot:
            print("ReadPulse failed:",  e)
            plt.figure(figsize=(6, 3))
            plt.plot(pulse)
            plt.title(f"FAILED: {path}")
            plt.show()

        return [np.nan, np.nan, np.nan, np.nan]


def ReadPulse(Data_path, pulse, target="Pulse_ms", path="", debug_plot=False):
    with open(f"{Data_path}/input.json") as f:
        para = json.load(f)

    if np.mean(pulse) <= 0:
        pulse *= -1

    pulse_filt = gp.BesselFilter(pulse, para["rate"], para["cutoff"]) if target != "Pulse_ms" else pulse

    if not np.all(np.isfinite(pulse_filt)):
        print(f"ReadPulse failed: non-finite values in pulse ({path})")
        save_readpulse_debug(Data_path, pulse, pulse_filt, para, path, "non-finite values")
        return [np.nan, np.nan, np.nan, np.nan]

    peak = np.max(pulse_filt)
    peak_index = int(np.argmax(pulse_filt))

    if peak_index < 10:
        print(f"ReadPulse failed: peak too early ({path})")
        save_readpulse_debug(Data_path, pulse, pulse_filt, para, path, "peak too early", peak_index=peak_index)
        return [np.nan, np.nan, np.nan, np.nan]

    l = max(0, peak_index - 10)
    r = min(len(pulse_filt), peak_index + 90)
    peak_av = np.mean(pulse_filt[l:r])

    rise_90 = None
    for i in reversed(range(0, peak_index)):
        if pulse_filt[i] <= peak * 0.9:
            rise_90 = i
            break
    if rise_90 is None:
        print(f"ReadPulse failed: rise_90 not found ({path})")
        save_readpulse_debug(Data_path, pulse, pulse_filt, para, path, "rise_90 not found", peak_index=peak_index)
        return [np.nan, np.nan, np.nan, np.nan]

    rise_10 = None
    for j in reversed(range(0, rise_90)):
        if pulse_filt[j] <= peak * 0.1:
            rise_10 = j
            break
    if rise_10 is None:
        print(f"ReadPulse failed: rise_10 not found ({path})")
        save_readpulse_debug(Data_path, pulse, pulse_filt, para, path, "rise_10 not found", peak_index=peak_index, rise_90=rise_90)
        return [np.nan, np.nan, np.nan, np.nan]

    rise = (rise_90 - rise_10) / para["rate"]

    st_l = para["SettlingTime"] - 10
    st_r = para["SettlingTime"] + 90
    if st_l < 0 or st_r > len(pulse_filt) or st_l >= st_r:
        print(f"ReadPulse failed: invalid settling window ({path})")
        save_readpulse_debug(Data_path, pulse, pulse_filt, para, path, "invalid settling window", peak_index=peak_index, rise_10=rise_10, rise_90=rise_90)
        return [np.nan, np.nan, np.nan, np.nan]

    ST_window = pulse_filt[st_l:st_r]
    if len(ST_window) == 0 or not np.all(np.isfinite(ST_window)):
        print(f"ReadPulse failed: invalid settling samples ({path})")
        save_readpulse_debug(Data_path, pulse, pulse_filt, para, path, "invalid settling samples", peak_index=peak_index, rise_10=rise_10, rise_90=rise_90)
        return [np.nan, np.nan, np.nan, np.nan]

    ST_height = np.mean(ST_window)

    if debug_plot:
        t = np.arange(len(pulse_filt)) / para["rate"]
        plt.figure(figsize=(8, 4))
        plt.plot(t, pulse_filt, label="pulse")
        plt.axvline(t[peak_index], color="r", linestyle="--", label="peak")
        plt.axvline(t[rise_10], color="g", linestyle="--", label="10%")
        plt.axvline(t[rise_90], color="orange", linestyle="--", label="90%")
        plt.axhline(peak * 0.1, color="g", alpha=0.3)
        plt.axhline(peak * 0.9, color="orange", alpha=0.3)
        plt.axvspan(st_l / para["rate"], st_r / para["rate"], color="gray", alpha=0.2, label="Settling window")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return [peak_av, peak_index, rise, ST_height]


def MakeOutput(Data_path,target):
    with open(f"{Data_path}/input.json") as f:
        para = json.load(f)

    if target != "Pulse_ms":
        print("BesselFilter")
    for i, posi in enumerate(para["position"]):
        print(f"{i+1}/{len(para['position'])}")
        for ch in[0,1]:
            print(f"Ch:{ch}")
            results=[]
            pulse_numbers=[]
            pulse_pathes = natsort.natsorted(glob.glob(f'{Data_path}/{para["E"]}keV_{posi}/{target}/CH{ch}/CH{ch}_*.dat'))

            for path in tqdm.tqdm(pulse_pathes):
                pattern = fr'CH{ch}_(\d+).dat'
                match = re.search(pattern, path)
                pulse_numbers.append(match.group(1))

                pulse=np.loadtxt(path)
                result=ReadPulse(Data_path,pulse,path=path,target=target)
                results.append(result)

            columns=["height","peak_index","rise","ST_Height"]
            df = pd.DataFrame(results,columns=columns,index=pulse_numbers)
            df.index.name = "id"
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
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return np.nan, np.nan

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
        
        mask = (
            np.isfinite(CH0_heights)
            & np.isfinite(CH1_heights)
            & (CH1_heights != 0)
        )
        ratios = CH0_heights[mask] / CH1_heights[mask]

        positions=find_nearest_values(ratios,fit_para)

        positions=spline(ratios)
        positions = np.asarray(positions, dtype=float)
        positions = positions[np.isfinite(positions)]
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

    colors=generate_symmetric_colors(len(fwhms))

    ene_reso_sum=[]

    for CH0,CH1,posi,color in zip(CH0_files,CH1_files,para["position"],colors):
        CH0_heights=ReadOutput(CH0,"height")
        CH1_heights=ReadOutput(CH1,"height")
        heights=CH0_heights+CH1_heights
        fwhm,reso=MakeHistgram(heights,posi,color)
        ene_reso_sum.append(reso*para["E"])

    plt.xlabel("Current[A]",fontsize=15)
    plt.ylabel("Count",fontsize=15)
    plt.tight_layout()
    plt.legend(labelspacing=0,fontsize=8, markerscale=0.5)
    plt.savefig(f"{Data_path}/energy_sum_histgram_{target}_{para["E"]}.png")
    if show:
        plt.show()
        plt.plot(para["position"],ene_reso_sum)
        plt.show()
    else:
        plt.clf()

    ene_reso_max=[]

    for CH0,CH1,posi,color in zip(CH0_files,CH1_files,para["position"],colors):
        CH0_heights=ReadOutput(CH0,"height")
        CH1_heights=ReadOutput(CH1,"height")
        heights=np.maximum(CH0_heights,CH1_heights)
        fwhm,reso=MakeHistgram(heights,posi,color)
        ene_reso_max.append(reso*para["E"])

    plt.xlabel("Current[A]",fontsize=15)
    plt.ylabel("Count",fontsize=15)
    plt.tight_layout()
    plt.xlim(0,None)
    plt.legend(labelspacing=0,fontsize=8, markerscale=0.5)
    plt.savefig(f"{Data_path}/energy_max_histgram_{target}_{para["E"]}.png")
    if show:
        plt.show()
        plt.plot(para["position"],ene_reso_max)
        plt.show()
    else:
        plt.clf()

    ene_reso_min=[]

    for CH0,CH1,posi,color in zip(CH0_files,CH1_files,para["position"],colors):
        CH0_heights=ReadOutput(CH0,"height")
        CH1_heights=ReadOutput(CH1,"height")
        heights=np.minimum(CH0_heights,CH1_heights)
        fwhm,reso=MakeHistgram(heights,posi,color)
        ene_reso_min.append(reso*para["E"])

    plt.xlabel("Current[A]",fontsize=15)
    plt.ylabel("Count",fontsize=15)
    plt.tight_layout()
    plt.legend(labelspacing=0,fontsize=8, markerscale=0.5)
    plt.savefig(f"{Data_path}/energy_min_histgram_{target}_{para["E"]}.png")
    if show:
        plt.show()
        plt.plot(para["position"],ene_reso_min)
        plt.show()
    else:
        plt.clf()

    ene_reso_st=[]

    for CH0,CH1,posi,color in zip(CH0_files,CH1_files,para["position"],colors):
        CH0_heights=ReadOutput(CH0,"ST_Height")
        CH1_heights=ReadOutput(CH1,"ST_Height")
        heights=CH0_heights+CH1_heights
        fwhm,reso=MakeHistgram(heights,posi,color)
        ene_reso_st.append(reso*para["E"])

    plt.xlabel("Current[A]",fontsize=15)
    plt.ylabel("Count",fontsize=15)
    plt.tight_layout()
    plt.legend(labelspacing=0,fontsize=8, markerscale=0.5)
    plt.savefig(f"{Data_path}/energy_ch_histgram_{target}_{para["E"]}.png")
    if show:
        plt.show()
        plt.plot(para["position"],ene_reso_st)
        plt.show()
    else:
        plt.clf()

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

def try_ReadPulse(Data_path):
    pulse_path=input("Pulse path:")
    pulse=np.loadtxt(pulse_path)
    peak_av, peak_index, rise, ST_height=ReadPulse(Data_path,pulse,path=pulse_path,debug_plot=True)
    print(f"peak_av: {peak_av}, peak_index: {peak_index}, rise: {rise}, ST_height: {ST_height}")


show=False
Data_path="H:\\hata\\1332_142_136_300split"
MakeOutput(Data_path,"Pulse_noise")
#MakeOutput(Data_path,"Pulse_ms")
#MakeOutput(Data_path,"Pulse_ms_noise")
#MakeOutput(Data_path,"pulse_noise_ms_test")
Resos(Data_path,"Pulse_noise",show)
#Resos(Data_path,"Pulse_ms",show)
#Resos(Data_path,"Pulse_ms_noise",show)
#Resos(Data_path,"pulse_noise_ms_test",show)
#try_ReadPulse(Data_path)
