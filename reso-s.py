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
import warnings
import os

warnings.filterwarnings('error')

def ReadOutput(FilePath,key,filter_path=None):
    df=pd.read_csv(FilePath)
    if filter_path is not None:
        if os.path.exists(filter_path):
            filter=np.loadtxt(filter_path)
            df = df[df['id'].isin(filter)]
    data = df[key].to_numpy()
    return data

def remove_outliers(data, percentile=0.1):
    lower_bound = np.percentile(data, percentile)
    upper_bound = np.percentile(data, 100 - percentile)
    return data[(data >= lower_bound) & (data <= upper_bound)]

def ReadPulse(Data_path,pulse,path,target):
    with open(f"{Data_path}/input.json") as f:
        para=json.load(f)
    pulse*=-1
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
        return None

def LookPulse(Data_path,target,selected=None):
    with open(f"{Data_path}/input.json") as f:
        para = json.load(f)
    if not selected:
        pulse_path = natsort.natsorted(glob.glob(f'{Data_path}/{para["E"]}keV_{para["position"][0]}/{target}/CH0/CH0_*.dat'))[0]
    else:
        pulse_path=f'{Data_path}/{para["E"]}keV_{para["position"][0]}/{target}/CH0/CH0_{selected}.dat'
    pulse=np.loadtxt(pulse_path)
    pulse=gp.BesselFilter(pulse,para["rate"],para["cutoff"])
    pulse*=-1
    time=np.arange(para["samples"])*(1/para["rate"])
    plt.plot(time,pulse,label="Pulse")
    try:
        peak = np.max(pulse)
        peak_index = np.argmax(pulse)

        # y軸の範囲を取得
        ymin, ymax = plt.ylim()

        # x=1からx=3まで、y軸全体を塗りつぶす
        plt.fill_betweenx([ymin, ymax], x1=time[peak_index - 10], x2=time[peak_index +90], color='lightblue', alpha=0.5)

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

        plt.fill_betweenx([ymin, ymax], x1=time[rise_10], x2=time[rise_90], color='lightgreen', alpha=0.5)

    except:
        print("Err")
    plt.show()


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
                pulse=np.loadtxt(path)
                result=ReadPulse(Data_path,pulse,path,target)
                if result is not None:
                    result=[match.group(1)]+result
                    results.append(result)

            columns=["id","height","peak_index","rise","ST_Height"]
            df = pd.DataFrame(results,columns=columns)
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

def MakeHistgram_without_plot(data,posi,HistColor=None):
    bin_num = optimal_bin_count(data)
    hist, bin_edges = np.histogram(data, bins=bin_num, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # ビンの中心を計算
    initial_guess = [np.max(hist), np.mean(data), np.std(data)]

    # ガウスフィッティング（標準偏差が負にならないよう制約を加える）
    try:
        popt, _ = curve_fit(
            gaussian, bin_centers, hist, p0=initial_guess, maxfev=1000000,
            bounds=([0, min(data), 0], [np.inf, max(data), np.inf])  # 標準偏差の下限を0に制限
        )
    except RuntimeError:
        print(f"Warning: Gaussian fitting failed for posi {posi}")
        return None, None  # フィッティング失敗時の処理

    amp_fit, mean_fit, stddev_fit = popt
    fwhm = 2 * stddev_fit * np.sqrt(2 * np.log(2))

    # 標準偏差が0に近すぎる場合、FWHMの計算が不安定になるためチェック
    if stddev_fit <= 0:
        print(f"Warning: Invalid standard deviation ({stddev_fit}) for posi {posi}")
        return None, None  # 異常値を回避

    return fwhm, fwhm / mean_fit

def MakeHistgram(data, posi, HistColor=None):
    bin_num = optimal_bin_count(data)
    hist, bin_edges = np.histogram(data, bins=bin_num, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # ビンの中心を計算
    initial_guess = [np.max(hist), np.mean(data), np.std(data)]

    if HistColor is not None:
        plt.hist(data, bins=bin_num, density=False, label=f"abs-{posi}", color=HistColor)
    else:
        plt.hist(data, bins=bin_num, density=False, label=f"abs-{posi}")

    # ガウスフィッティング（標準偏差が負にならないよう制約を加える）
    try:
        popt, _ = curve_fit(
            gaussian, bin_centers, hist, p0=initial_guess, maxfev=1000000,
            bounds=([0, min(data), 0], [np.inf, max(data), np.inf])  # 標準偏差の下限を0に制限
        )
    except RuntimeError:
        print(f"Warning: Gaussian fitting failed for posi {posi}")
        return None, None  # フィッティング失敗時の処理

    amp_fit, mean_fit, stddev_fit = popt
    fwhm = 2 * stddev_fit * np.sqrt(2 * np.log(2))

    # 標準偏差が0に近すぎる場合、FWHMの計算が不安定になるためチェック
    if stddev_fit <= 0:
        print(f"Warning: Invalid standard deviation ({stddev_fit}) for posi {posi}")
        return None, None  # 異常値を回避

    # フィッティング曲線の描画
    x_fit = np.linspace(bin_edges[0], bin_edges[-1], 1000)
    plt.plot(x_fit, gaussian(x_fit, *popt), color="red", alpha=0.5)

    return fwhm, fwhm / mean_fit

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



def Resos(Data_path, target, show):
    fit_para_path = f"{Data_path}/ratios.txt"
    fit_para = np.loadtxt(fit_para_path, delimiter=',')
    with open(f"{Data_path}/input.json", "r") as f:
        para = json.load(f)

    CH0_files = [f"{Data_path}/{para['E']}keV_{posi}/{target}/output_TES0.csv" for posi in para["position"]]
    CH1_files = [f"{Data_path}/{para['E']}keV_{posi}/{target}/output_TES1.csv" for posi in para["position"]]

    # spline 準備
    ratio_base, posi_base = fit_para[:,1], fit_para[:,0]
    sorted_indices = np.argsort(ratio_base)
    spline = UnivariateSpline(ratio_base[sorted_indices], posi_base[sorted_indices], k=3, s=0)

    colors = generate_symmetric_colors(len(para["position"]))
    
    # ----------------------------------------------------
    # ⭐ データを最初にまとめて読み込み
    all_data = []
    for CH0, CH1, posi in zip(CH0_files, CH1_files, para["position"]):
        selected_ids_path = f"{Data_path}/{para['E']}keV_{posi}/{target}/selected_ids.txt"
        data = {
            "posi": posi,
            "CH0_height": ReadOutput(CH0, "height", selected_ids_path),
            "CH1_height": ReadOutput(CH1, "height", selected_ids_path),
            "CH0_st": ReadOutput(CH0, "ST_Height", selected_ids_path),
            "CH1_st": ReadOutput(CH1, "ST_Height", selected_ids_path),
        }
        all_data.append(data)
    # ----------------------------------------------------
    # 位置推定ヒストグラム
    fwhms = []
    for data in all_data:
        ratios = data["CH0_height"] / data["CH1_height"]
        positions = spline(ratios)
        fwhm, _ = MakeHistgram(positions, data["posi"])
        fwhms.append(fwhm)

    np.savetxt(f"{Data_path}/fwhms_{target}.txt", fwhms)
    plt.xlabel("Position", fontsize=15)
    plt.ylabel("Count", fontsize=15)
    plt.title(f"{para['E']}keV")
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"{Data_path}/position_histgram_{target}_{para['E']}.png")
    if show:
        plt.show()
        plt.plot(para["position"], fwhms)
        plt.show()
    else:
        plt.clf()

    # ----------------------------------------------------
    # Sum / Max / Min （すべて height で計算）
    
    results = {}

    for mode_name, combine_func in [("Sum", lambda c0, c1: c0 + c1),
                                    ("Max", lambda c0, c1: np.maximum(c0, c1)),
                                    ("Min", lambda c0, c1: np.minimum(c0, c1))]:
        ene_reso_list = []
        for data, color in zip(all_data, colors):
            heights = combine_func(data["CH0_height"], data["CH1_height"])
            fwhm, reso = MakeHistgram(heights, data["posi"], color)
            ene_reso_list.append(reso * para["E"])
        results[mode_name] = ene_reso_list

        plt.xlabel("Current[A]", fontsize=15)
        plt.ylabel("Count", fontsize=15)
        plt.tight_layout()
        plt.legend(labelspacing=0, fontsize=8, markerscale=0.5)
        plt.savefig(f"{Data_path}/energy_{mode_name.lower()}_histgram_{target}_{para['E']}.png")
        if show:
            plt.show()
            plt.plot(para["position"], ene_reso_list)
            plt.show()
        else:
            plt.clf()

    # ----------------------------------------------------
    # ST_Height（別枠で）
    ene_reso_st = []
    for data, color in zip(all_data, colors):
        heights = data["CH0_st"] + data["CH1_st"]
        fwhm, reso = MakeHistgram_without_plot(heights, data["posi"], color)
        ene_reso_st.append(reso * para["E"])
    results["ST"] = ene_reso_st

    if False:
        plt.xlabel("Current[A]", fontsize=15)
        plt.ylabel("Count", fontsize=15)
        plt.tight_layout()
        plt.legend(labelspacing=0, fontsize=8, markerscale=0.5)
        plt.savefig(f"{Data_path}/energy_st_histgram_{target}_{para['E']}.png")
        if show:
            plt.show()
            plt.plot(para["position"], ene_reso_st)
            plt.show()
        else:
            plt.clf()

    # ----------------------------------------------------
    # CSV 出力
    index_label = (np.array(para["position"]) - 0.5) * para["length"] / para["n_abs"]
    df = pd.DataFrame(results, index=index_label)
    df.to_csv(f"{Data_path}/ene_resos_{target}.csv")


show=False
Data_path="H:/hata2025/1332_120_100"
#LookPulse(Data_path,"Pulse_ms",selected=209809)
#MakeOutput(Data_path,"Pulse_noise")
#MakeOutput(Data_path,"Pulse_ms")
#MakeOutput(Data_path,"Pulse_ms_noise")
#MakeOutput(Data_path,"pulse_noise_ms_test")
#Resos(Data_path,"Pulse_noise",True)
Resos(Data_path,"Pulse_ms",show)
#Resos(Data_path,"Pulse_ms_noise",show)
#Resos(Data_path,"pulse_noise_ms_test",show)