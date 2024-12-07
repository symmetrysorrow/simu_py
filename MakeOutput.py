#2024/12/04 Hata&tagawa

import numpy as np
import pandas as pd
import glob
from natsort import natsorted
import json
import re

Data_path="d:/tagawa/20241203/room1_ch1ch2_220mK_920uA920uA_gain5_trig0.2V_rate500k_samples100k"


def ReadPulse(pulse,path):
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

def NormalOutput(target):
    print("normal")
    with open(f"{Data_path}/input.json") as f:
        input = json.load(f)
    for posi in input["Position"]:
        for ch in[0,1]:
            results=[]
            pulse_numbers=[]
            pulse_pathes = natsorted(glob.glob(f'{Data_path}/{input["E"]}keV_{posi}/{target}/CH{ch}/CH{ch}_*.dat'))

            for path in pulse_pathes:
                pattern = fr'CH{ch}_(\d+).dat'
                match = re.search(pattern, path)
                pulse_numbers.append(match.group(1))

                pulse=np.loadtxt(path)
                results.append(ReadPulse(pulse))

            columns=["height","peak_index","rise","CheckPointHeight"]
            df = pd.DataFrame(results,columns=columns,index=pulse_numbers)
            df.to_csv(f"{Data_path}/{input["E"]}keV_{posi}/{target}/output_TES{ch}.csv")    

def Posi_reso(target,bin_num):
    print()
    with open(f"{Data_path}/input.json") as f:
        input = json.load(f)
    ratios=np.loadtxt(f"{Data_path}/ratios.txt")
    for posi in input["Position"]:
        data_ch0 = pd.read_csv(f"{Data_path}/{input["E"]}keV_{posi}/{target}/output_TES0.csv")
        heights_ch0=data_ch0["height"].to_numpy()
        data_ch0 = pd.read_csv(f"{Data_path}/{input["E"]}keV_{posi}/{target}/output_TES1.csv")
        heights_ch1=data_ch0["height"].to_numpy()

NormalOutput("Pulse_ms")
NormalOutput("Pulse_ms_noise")
NormalOutput("Pulse_noise")
