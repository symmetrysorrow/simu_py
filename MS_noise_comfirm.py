# -*- coding: utf-8 -*-

# --------last updated 2018/12/01 by kurume-------------------

import math
import ctypes
import shutil
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import matplotlib.cm as cm
import pandas as pd
import json
import os
import re
import glob
import tqdm
import random
import scipy.fftpack as sf
import scipy.fftpack as fft
import cmath
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import general
import tqdm
import random
# --------------------------------------------------------------
k_b = 1.381 * 1.0e-23  # Boltzmann's constant
ptfn_Flink = 0.5
e = 1.602e-19 
eta = 100
amp = 10
cf = 10000
zure = 30 

pulse_num=500

output="H:/hata2025/1332_noise&MS"
ENERGY_CANDIDATES = [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300]

def MakeNoiseAndPulse():
    with open(f'{output}/input.json', "r") as f:
        para = json.load(f)
    noise_spe_dens = np.loadtxt(f"{output}/noise_total.dat")
    sample=int(para['samples'])
    df=para['rate']/sample
    d_length=sample
    noise_spe_dens*=np.sqrt(df)*(d_length/np.sqrt(2))*2
    amplitude_model = np.zeros(sample)
    pulse=np.loadtxt(f"{output}/{para["E"]}keV_{para["position"][0]}/pulse/CH0/CH0_1.dat")
    for i in tqdm.tqdm(range(500)):
        random_coeff=np.random.rand()
        noise_time=general.GN(noise_spe_dens)[:sample]
        noise_time+=random_coeff*pulse[20000:20000+sample]
        noise_time=general.Bessel(noise_time,para['rate'],10000)
        noise_time=general.Bessel(noise_time,para['rate'],para["cutoff"])
        noise_fft=sf.fft(noise_time)
        noise_amp = np.abs(noise_fft)
        amplitude_model += noise_amp
    amplitude_model /= 100
    df=para["rate"]/sample
    power=amplitude_model**2 / df
    amp_dens=np.sqrt(power)
    amp_dens = amp_dens[: int(sample / 2) + 1] * eta * 1e+6
    plt.plot(amp_dens)
    plt.xlabel("Frequency [Hz]", fontsize=20)
    plt.ylabel("Amplitude [uA/rtHz]", fontsize=20)
    plt.loglog()
    plt.savefig(f"{output}/noise_total-bessel100k-with-pulseTail.png", dpi=350)
    plt.clf()
    np.savetxt(f"{output}/noise_total-bessel100k-with-pulseTail.dat",amp_dens)

def ShowPulse():
    with open(f"{output}/input.json", "r") as f:
        para = json.load(f)
    time=general.GetTime(para["rate"],para["samples"])
    for ene in ENERGY_CANDIDATES:
        Pulse=np.loadtxt(f"{output}/pulse/CH1_ene={ene}.dat")
        plt.plot(time,Pulse,label=f"{ene}keV")
    plt.legend()
    plt.show()


MakeNoiseAndPulse()
