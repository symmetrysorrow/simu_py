# -*- coding: utf-8 -*-

# --------last updated 2018/12/01 by kurume-------------------

import math
import ctypes
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import matplotlib.cm as cm
import pandas as pd
import json
import os
import glob
import random
import getpara as gp
from natsort import natsorted
from scipy.optimize import curve_fit
import scipy.fftpack as sf
import scipy.fftpack as fft
import cmath
from concurrent.futures import ThreadPoolExecutor
# --------------------------------------------------------------
k_b = 1.381 * 1.0e-23  # Boltzmann's constant
ptfn_Flink = 0.5
e = 1.602e-19 
eta = 100
amp = 10
cf = 10000
zure = 30 

pulse_num=200

output="./output/662_142_136"

def random_noise(spe, seed):
    spe_re = spe[::-1]  # reverce
    spe_mirror = np.r_[spe, spe_re]
    np.random.seed(seed)
    phase = (2 * np.pi - 0) * np.random.rand(len(spe_mirror))  # random phase
    complex = [cmath.rect(i, j) for i, j in zip(spe_mirror, phase)]
    complex_con = [i.conjugate() for i in complex[len(spe) :]]  # conjugate
    return np.r_[complex[: len(spe)], complex_con]

def MakePulse():
    with open(f"{output}/input.json", "r") as f:
        para = json.load(f)
    
    makepulse_dll = ctypes.CDLL('./MakePulse.dll')

    makepulse_dll.MakePulse.argtypes = [ctypes.c_char_p]
    makepulse_dll.MakePulse.restype = None

    input_string = f"{output}"
    makepulse_dll.MakePulse(input_string.encode('utf-8'))

def FitRatios():
    term=15

    with open(f"{output}/input.json", "r") as f:
        para = json.load(f)

    all_position= list(range(1, para['n_abs'] + 1))

    time = np.linspace(0, para["samples"] / para["rate"], int(para["samples"]))

    cnt=0

    for posi in para["position"]:
        data = np.loadtxt(f"{output}/{para["E"]}keV_{posi}/pulse/CH0/CH0_{posi}.dat")
        plt.plot(time*1e3,data*1e6,label=f"abs-{posi}",color=cm.hsv(float(cnt) / float(len(para["position"]))))
        cnt+=1

    plt.xlabel("Time [ms]")
    plt.ylabel("Current [uA]")
    plt.xlim(0,10)
    plt.ylim(0,1)
    plt.grid()
    plt.legend(loc="best", fancybox=True, shadow=True, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{output}/pulse_post.png", dpi=700)
    plt.show()
    plt.cla()
    
    heights_ch0=[]
    heights_ch1=[]
    rises_ch0=[]
    rises_ch1=[]
    decays_ch0=[]
    decays_ch1=[]

    for j in [0,1]:
        pulses = natsorted(glob.glob(f"{output}/{para["E"]}keV_*/pulse/CH{j}/CH{j}_*.dat"))
        data_array = []
        for i in pulses:
            pixel = gp.num(os.path.basename(i))[1]
            data = np.loadtxt(i) 
            peak,peak_av,peak_index = gp.peak(data,0,int(para['samples']),10,100)
            rise, rise_10, rise_90 = gp.risetime(data, peak_av, np.argmax(data),0.9,0.1,para['rate'])
            decay, decay_10, decay_90 = gp.decaytime(data, peak, np.argmax(data),0.9,0.7, para['rate'])
            if j==0:
                heights_ch0.append(peak)
                rises_ch0.append(rise)
                decays_ch0.append(decay)
            if j==1:
                heights_ch1.append(peak)
                rises_ch1.append(rise)
                decays_ch1.append(decay)
    
    # height/height
    heights_ch0=np.array(heights_ch0)
    heights_ch1=np.array(heights_ch1)
    rises_ch0=np.array(rises_ch0)
    rises_ch1=np.array(rises_ch1)
    decays_ch0=np.array(decays_ch0)
    decays_ch1=np.array(decays_ch1)

    ratio = heights_ch0 / heights_ch1

    position = (
        (np.array(all_position)  -  1/2 ) * para["length"] / para['n_abs']
    )
    popt, pcov = curve_fit(gp.multi_func, ratio, position, p0=np.zeros(term + 1))

    np.savetxt(f"{output}/fit_para.txt", popt)
    x_fit = np.arange(np.min(ratio), np.max(ratio), 0.01)
    y_fit = gp.multi_func(x_fit, *tuple(popt))


    # --- Plot --------------------------------
    value = [(float(i)) / float(len(position) + 1) for i in range(len(position))]

    #plt.scatter(df["rise"] * 1e3, df["height"] * 1e6, c=value, s=3.0,cmap="hsv")
    plt.scatter(rises_ch0 * 1e3, heights_ch0 * 1e6, c=value,cmap="hsv")
    # plt.title("risetime vs pulse height")
    plt.xlabel("risetime [ms]")
    plt.ylabel("pulseheight [uA]")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{output}/rise_height.png", dpi=350)
    plt.show()

    plt.scatter(decays_ch0 * 1e3, heights_ch0 * 1e6,c=value, cmap="hsv")
    # plt.title("dcaytime vs pulse height")
    plt.xlabel("decaytime [ms]")
    plt.ylabel("pulseheight [uA]")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{output}/decay_height.png", dpi=350)
    plt.show()

    plt.scatter(heights_ch0* 1e6, heights_ch1 * 1e6,c=value, cmap="hsv")
    # plt.title("risetime vs pulse height")
    plt.xlabel("pulseheight (CH0) [uA]")
    plt.ylabel("pulseheight (CH1) [uA]")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{output}/height1_height2.png", dpi=350)
    plt.show()

    plt.scatter(ratio, position, c=value,cmap="hsv")
    plt.plot(x_fit, y_fit, "--")
    # plt.title("pixel vs pulseheight1/ pulseheight2")
    plt.xlabel("pulseheight (CH0)/ pulseheight (CH1)")
    plt.ylabel("position [mm]")
    plt.ylim(0-1,para['length']+1)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{output}/height_ratio.png", dpi=350)
    plt.show()

def MakeNoise():
    def create_output_directry(*path):
        if path != None:
        #
            filenumber = path[0]
        else:
            for i in range(1000):
                folder = os.path.exists(f"./output/{str(i+1)}")
                if not folder:
                    os.makedirs(f"./output/{str(i+1)}", exist_ok=True)
                    filenumber = i + 1
                    break
                else:
                    filenumber = 0
                    continue
        return str(filenumber)

    # add omega at diagonal
    def add_omega(M, n_abs, omega):
        omega_list = np.full(n_abs + 4, omega * 1.0j, dtype=np.complex128)
        omega_diag = np.diag(omega_list)
        return M + omega_diag
    
    with open (f"{output}/input.json", "r") as f:
        para = json.load(f)
        
    output_number = create_output_directry(para["output"])

    n_abs = para["n_abs"]  # absorber pixel
    C_abs = para["C_abs"] / n_abs  # heat capacity per 1-pixel
    C_tes = para["C_tes"]  # heat capacity (TES)
    G_abs_abs = float(para["G_abs-abs"]) * (
        n_abs - 1
    )  # thermal conductivity per 1-block
    G_abs_tes = para["G_abs-tes"]  # thermal conductivity (absorber-TES)
    G_tes_bath = para["G_tes-bath"]  # thermal conductivity (TES-bath)
    R = para["R"]  # R_TES
    R_l = para["R_l"]  # R_load (shunt)
    T_c = para["T_c"]  # T_c
    T_bath = para["T_bath"]  # T_bath
    a = para["alpha"]  # alpha
    b = para["beta"]  # beta
    L = para["L"]  # Indactance
    n = para["n"]  # dimensionless constant (dominant thermal transport mechanism )
    E = para["E"]  # energy
    length = para["length"]  # length
    rate = int(para["rate"])  # sample rate
    samples = int(para["samples"])  # samples
    para["output"] = output

    time = np.linspace(0, samples / rate, samples)
    frequency = np.arange(0, rate, rate / samples)

    I = np.sqrt((G_tes_bath * T_c * (1 - ((T_bath / T_c) ** n))) / (n * R))  # I_tes

    t_el = L / (R_l + R * (1 + b))  # tau_electron
    L_I = (a * (I**2) * R) / (G_tes_bath * T_c)  # Loop gain
    t_I = C_tes / ((1 - L_I) * G_tes_bath)  # tau_?

    # ----- Noises -----------------------------------------
    # Thremal Fluction Noise
    ptfn_tes_bath = np.sqrt(
        4 * k_b * T_c**2 * G_tes_bath * ptfn_Flink
    )  # Phonon Noise (tes-bath) [W/√Hz]
    ptfn_abs_tes = np.sqrt(
        4 * k_b * T_c**2 * G_abs_tes * ptfn_Flink
    )  # Phonon Noise (abs-tes) [W/√Hz]
    ptfn_abs_abs = np.sqrt(
        4 * k_b * T_c**2 * G_abs_abs * ptfn_Flink
    )  # Phonon Noise (abs-abs) [W/√Hz]

    # Johnson Noise
    enj = np.sqrt(4 * k_b * T_c * R * (1 + 2 * b + b**2))  # at TES
    enj_R = np.sqrt(4 * k_b * T_bath * R_l)  # at R_l

    # noise sources matrix N
    def matrix_N(n_abs):
        X = np.zeros((n_abs + 7, n_abs + 4), dtype=np.complex128)  # initialize matrix
        for i in range(n_abs + 7):
            if i == 0:  # johnson Noise (TES1)
                for j in range(n_abs + 4):
                    if j == 0:
                        X[i, j] = -enj / L
                    elif j == 1:
                        X[i, j] = I * enj / C_tes

            elif i == 1:  # johnson Noise (Load1)
                for j in range(n_abs + 4):
                    if j == 0:
                        X[i, j] = enj_R / L
                        break

            elif i == 2:  # Phonon Noise (TES1-Bath)
                for j in range(n_abs + 4):
                    if j == 1:
                        X[i, j] = ptfn_tes_bath / C_tes
                        break

            elif i == 3:  # Phonon Noise (TES1-Absorber)
                for j in range(n_abs + 4):
                    if j == 1:
                        X[i, j] = ptfn_abs_tes / C_tes
                    elif j == 2:
                        X[i, j] = -ptfn_abs_tes / C_abs

            elif i == n_abs + 3:  # Phonon Noise (TES2-Absorber)
                for j in range(n_abs + 4):
                    if j == n_abs + 1:
                        X[i, j] = -ptfn_abs_tes / C_abs
                    elif j == n_abs + 2:
                        X[i, j] = ptfn_abs_tes / C_tes

            elif i == n_abs + 4:  # Phonon Noise (TES2-Bath)
                for j in range(n_abs + 4):
                    if j == n_abs + 2:
                        X[i, j] = ptfn_tes_bath / C_tes
                        break

            elif i == n_abs + 5:  # johnson Noise (Load2)
                for j in range(n_abs + 4):
                    if j == n_abs + 3:
                        X[i, j] = enj_R / L
                        break

            elif i == n_abs + 6:  # johnson Noise (TES2)
                for j in range(n_abs + 4):
                    if j == n_abs + 3:
                        X[i, j] = -enj / L
                    elif j == n_abs + 2:
                        X[i, j] = I * enj / C_tes

            else:  # Phonon Noise (Absorber-Absorber)
                for j in range(n_abs + 4):
                    if j == i - 2:
                        X[i, j] = ptfn_abs_abs / C_abs
                    elif j == i - 1:
                        X[i, j] = -ptfn_abs_abs / C_abs
        return X

    # --------------------------------------------------

    # matrix M without omega
    def matrix_M(n_abs, omega):
        X = np.zeros((n_abs + 4, n_abs + 4), dtype=np.complex128)  # initialize matrix
        for i in range(n_abs + 4):
            if i == 0:
                for j in range(n_abs + 4):
                    if j == 0:
                        X[i, j] = 1 / t_el + omega * 1.0j
                    elif j == 1:
                        X[i, j] = L_I * G_tes_bath / (I * L)
            elif i == 1:
                for j in range(n_abs + 4):
                    if j == 0:
                        X[i, j] = -I * R * (2 + b) / C_tes
                    elif j == 1:
                        X[i, j] = 1 / t_I + (G_abs_tes / C_tes) + omega * 1.0j
                    elif j == 2:
                        X[i, j] = -G_abs_tes / C_tes
            elif i == 2:
                for j in range(n_abs + 4):
                    if j == 1:
                        X[i, j] = -G_abs_tes / C_abs
                    elif j == 2:
                        X[i, j] = G_abs_tes / C_abs + G_abs_abs / C_abs + omega * 1.0j
                    elif j == 3:
                        X[i, j] = -G_abs_abs / C_abs
            elif i == n_abs + 1:
                for j in range(n_abs + 4):
                    if j == n_abs:
                        X[i, j] = -G_abs_abs / C_abs
                    elif j == n_abs + 1:
                        X[i, j] = (G_abs_tes + G_abs_abs) / C_abs + omega * 1.0j
                    elif j == n_abs + 2:
                        X[i, j] = -G_abs_tes / C_abs
            elif i == n_abs + 2:
                for j in range(n_abs + 4):
                    if j == n_abs + 1:
                        X[i, j] = -G_abs_tes / C_tes
                    elif j == n_abs + 2:
                        X[i, j] = 1 / t_I + (G_abs_tes / C_tes) + omega * 1.0j
                    elif j == n_abs + 3:
                        X[i, j] = -I * R * (2 + b) / C_tes
            elif i == n_abs + 3:
                for j in range(n_abs + 4):
                    if j == n_abs + 2:
                        X[i, j] = L_I * G_tes_bath / (I * L)
                    elif j == n_abs + 3:
                        X[i, j] = 1 / t_el + omega * 1.0j
            else:
                for j in range(n_abs + 4):
                    if j == i - 1:
                        X[i, j] = -G_abs_abs / C_abs
                    elif j == i:
                        X[i, j] = 2 * G_abs_abs / C_abs + omega * 1.0j
                    elif j == i + 1:
                        X[i, j] = -G_abs_abs / C_abs
        return X

    N = matrix_N(n_abs).T

    omega = frequency * 2 * math.pi
    noise = []

    cnt = 0
    for omg in omega:
        M = matrix_M(n_abs, omg)
        noise_out = np.abs(np.linalg.solve(M, N)[0])
        noise.append(noise_out)
    noise = np.array(noise).T
    noise = np.vstack([noise, np.sum(noise, axis=0)])

    np.savetxt(f"{output}/noise_spectral_total_alpha71beta1.6.dat", noise[n_abs + 7])

    np.savetxt(f"{output}/noise_spectral_absorber_alpha71beta1.6.dat",np.sum(noise[4 : n_abs + 3], axis=0))

    # --- grough Noise Spectral Density--------------------------------------------
    plt.figure(figsize=(8, 8))
    plt.plot(
        frequency,
        noise[0],
        color="red",
        linewidth=2,
        linestyle=(0, (5, 1)),
        label="Johnson Noise (TES1)",
    )
    plt.plot(
        frequency,
        noise[n_abs + 6],
        color="orange",
        linewidth=2,
        linestyle=(0, (5, 1)),
        label="Johnson Noise (TES2)",
    )
    plt.plot(
        frequency,
        noise[1],
        color="lawngreen",
        linewidth=2,
        linestyle=(0, (5, 5)),
        label="Johnson Noise (Load1)",
    )
    plt.plot(
        frequency,
        noise[n_abs + 5],
        color="greenyellow",
        linewidth=2,
        linestyle=(0, (5, 5)),
        label="Johnson Noise (Load2)",
    )
    plt.plot(
        frequency,
        noise[2],
        color="blue",
        linewidth=2,
        linestyle=(0, (3, 5, 1, 5)),
        label="Phonon Noise (TES1-Bath)",
    )
    plt.plot(
        frequency,
        noise[n_abs + 4],
        color="royalblue",
        linewidth=2,
        linestyle=(0, (3, 5, 1, 5)),
        label="Phonon Noise (TES2-Bath)",
    )
    plt.plot(
        frequency,
        noise[3],
        color="magenta",
        linewidth=2,
        linestyle=(0, (3, 1, 1, 1)),
        label="Phonon Noise (TES1-Absorber)",
    )
    plt.plot(
        frequency,
        noise[n_abs + 3],
        color="pink",
        linewidth=2,
        linestyle=(0, (3, 1, 1, 1)),
        label="Phonon Noise (TES2-Absorber)",
    )

    phonon_noise = np.sum(noise[4 : n_abs + 3], axis=0)

    plt.plot(
        frequency,
        phonon_noise,
        color="dodgerblue",
        linewidth=2,
        linestyle=(0, (3, 1, 1, 1, 1, 1)),
        label="Phonon Noise sum (Absorber-Absorber)",
    )

    plt.plot(
        frequency, noise[n_abs + 7], color="black", linewidth=3, label="Total Noise"
    )

    plt.xlabel("Frequency [Hz]", fontsize=20)
    plt.ylabel("Noise Spectral Density [A/rtHz]", fontsize=20)
    plt.ylim(10e-14, 10e-10)
    plt.xlim(10e-1, 10e5)
    plt.loglog()
    plt.grid()
    plt.legend(loc="best", fancybox=True, fontsize=10, ncol=2)
    plt.tight_layout()
    plt.savefig(f"{output}/noise_spectrum_alpha71beta1.6.png", dpi=700)
    plt.show()
    plt.cla()

def CheckPulse():
    with open(f'{output}/input.json', "r") as f:
        para = json.load(f)

    # simulated noise frequency domain
    noise_spe_dens = np.loadtxt(f"{output}/noise_spectral_total_alpha71beta1.6.dat")
    noise_samples = len(noise_spe_dens)

    time = np.linspace(0, para['samples'] / para['rate'], int(para['samples']))
    frequency_pulse = np.arange(0, para['rate'], para['rate'] / int(para['samples']))
    frequency_noise = np.arange(0, para['rate'], para['rate'] / noise_samples)

    # if list is empty, n_abs block output
    if para["position"] == []:
        para["position"] = list(range(1, para["n_abs"] + 1))
    
    # add random phase
    noise_spe = random_noise(noise_spe_dens, 0)
    # noise spectra random phase
    noise_spe_amp = np.abs(noise_spe)

    df = para['rate']/noise_samples
    df_time = para['rate']/int(para['samples'])
    dt = float(1/noise_samples)
    d_length = noise_samples

    ifft_input = noise_spe*np.sqrt(df)*(d_length/np.sqrt(2))*2 # *2 ???

    # simulated noise time domain
    noise_ifft= np.fft.ifft(ifft_input, noise_samples).real

    noise_fft = sf.fft(noise_ifft,int(noise_samples))[:int(noise_samples/2)]

    # simulated noise frequency domain (random phase)
    amp = np.abs(noise_fft)/np.sqrt(df)/(noise_samples/np.sqrt(2.))

    pulse = np.loadtxt(f"{output}/{para["E"]}keV_1/pulse/CH0/CH0_1.dat")

    pulse_fft = np.fft.fft(pulse)
    pulse_amp = np.abs(pulse_fft)/np.sqrt(df)/(noise_samples/np.sqrt(2.))

    pulse_noise = pulse + noise_ifft[: int(para["samples"])]
    pulse_noise_fft = sf.fft(pulse_noise,int(para["samples"]))
    pulse_noise_amp = np.abs(pulse_noise_fft)/np.sqrt(df_time)/(int(para["samples"])/np.sqrt(2.))

    # ---- Plot frequency domain ------
    plt.plot(frequency_noise[:int(noise_samples/2)],amp[:int(noise_samples/2)]*1e12,label = "ifft simulated fft (random phase)")
    plt.plot(frequency_noise[:int(noise_samples/2)],noise_spe_amp[:int(noise_samples/2)]*1e12,"o",markersize=3.0,label = "simulated noise (random phase)")
    plt.plot(frequency_noise[:int(noise_samples/2)],noise_spe_dens[:int(noise_samples/2)]*1e12,label = "simulated noise")
    plt.loglog()
    plt.xlabel("Frequency[Hz]")
    plt.ylabel("Intensity[pA/Hz$^{1/2}$]")
    plt.grid()
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output}/noise_spectra.png", dpi=350)
    plt.show()
    plt.cla()

    # ---- noise time domain --------
    plt.plot(time * 1e3,noise_ifft[:int(para['samples'])] * 1e6,linewidth=1.5)
    plt.xlabel("Time [ms]", fontsize=20)
    plt.ylabel("Current [uA]", fontsize=20)
    plt.grid()
    plt.tight_layout()
    #plt.legend(fontsize=12, loc='upper right')
    plt.savefig(f"{output}/noise_post.png", dpi=350)
    plt.show()
    plt.cla()

    # ---- pulse with noise time domain -------

    cnt = 0
    for i in para["position"]:
        data = np.loadtxt(f"{output}/{para["E"]}keV_{i}/pulse/CH0/CH0_{i}.dat")
        noise_spe = random_noise(noise_spe_dens, cnt)
        #noise = np.fft.ifft(noise_spe, noise_samples).real * 2
        ifft_input = noise_spe*np.sqrt(df)*(d_length/np.sqrt(2))*2 # *2 ???

        # simulated noise time domain
        noise_ifft= np.fft.ifft(ifft_input, noise_samples).real

            
        data += noise_ifft[: int(para["samples"])]
        #data = gp.BesselFilter(data,para['rate'],para['cutoff'])
            
        plt.plot(time * 1e3,data * 1e6,color=cm.hsv((float(cnt)) / float(len(para["position"]))),linewidth=1.5,label="abs" + str(i),)
        cnt += 1
    plt.xlabel("Time [ms]", fontsize=20)
    plt.ylabel("Current [uA]", fontsize=20)
    plt.xlim(0,10)
    plt.grid()
    plt.tight_layout()
    plt.legend(fontsize=10,loc="best", fancybox=True)
    plt.savefig(f"{output}/pulse_noise_post.png", dpi=350)
    plt.show()
    plt.cla()
        
        
    # ---- pulse with noise frequencyx domain -------

    plt.plot(frequency_pulse[1 : int(para['samples'] / 2)],pulse_noise_amp[1 : int(para['samples'] / 2)]*1e12,c="orange",label = 'pulse + noise (random phase)')
    plt.plot(frequency_noise[1 : int(para['samples'] / 2)],pulse_amp[1 : int(para['samples'] / 2)]*1e12+noise_spe_dens[1 : int(para['samples'] / 2)]*1e12,c="black",linewidth = 2,label = "pulse + noise")
    ## rawdata
    plt.plot(frequency_noise[1 : int(noise_samples / 2)],noise_spe_dens[1 : int(noise_samples / 2)]*1e12,"--",c="green",label = "noise")

    plt.plot(frequency_noise[1 : int(para['samples'] / 2)], pulse_amp[1 : int(para['samples'] / 2)]*1e12,"--",c="red",label= "pulse")

    plt.loglog()
    plt.xlabel("Frequency[Hz]")
    plt.ylabel("Intensity[pA/Hz$^{1/2}$]")
    plt.grid()
    plt.tight_layout()
    plt.legend(fontsize=10,loc="best", fancybox=True)
    plt.savefig(f"{output}/pulse_noise_spectra.png", dpi=350)
    plt.show()
    plt.cla()

def MultiPulse():
    with open(f"{output}/input.json", "r") as f:
        para = json.load(f)

    def AddPulse(output, para, i, j, noise_samples, noise_spe_dens):
        os.makedirs(f"{output}/{para['E']}keV_{i}/pulse_noise/CH{j}", exist_ok=True)
        data = np.loadtxt(f"{output}/{para['E']}keV_{i}/pulse/CH{j}/CH{j}_{i}.dat")

        df = para['rate'] / noise_samples
        d_length = noise_samples

        cnt = random.randint(1, 10000)

        for k in range(pulse_num):
            noise_spe = random_noise(noise_spe_dens, cnt)
            ifft_input = noise_spe * np.sqrt(df) * (d_length / np.sqrt(2)) * 2 
            noise_ifft = np.fft.ifft(ifft_input, noise_samples).real
            data_n = data + noise_ifft[:len(data)]
            data_n = gp.BesselFilter(data_n, para['rate'], para["cutoff"])
            np.savetxt(f"{output}/{para['E']}keV_{i}/pulse_noise/CH{j}/CH{j}_{k}.dat", data_n)

    noise_spe_dens = np.loadtxt(f"{output}/noise_spectral_total_alpha71beta1.6.dat")
    noise_samples = len(noise_spe_dens)

    for j in [0,1]:
         with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(AddPulse, output, para, i, j, noise_samples, noise_spe_dens)
                for i in para["position"]
            ]
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"Error occurred: {e}")

def MakeHistgram():
    def multi_func(X, *params):
        Y = np.zeros_like(X)
        for i, param in enumerate(params):
            Y += param * X**i
        return Y
    def gaussian(x, amp, mean, stddev):
        return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

    with open(f"{output}/input.json", "r") as f:
        para = json.load(f)

    bin_num=20

    fit_para=np.loadtxt(f"{output}/fit_para.txt")

    fwhms=[]

    cnt=0

    plt.figure(figsize=(10,5))

    for i in para["position"]:
        data_0=pd.read_csv(f"{output}/{para["E"]}keV_{i}/Pulse_noise/output_TES0.csv")
        data_1=pd.read_csv(f"{output}/{para["E"]}keV_{i}/Pulse_noise/output_TES1.csv")
        ratio = data_0['height'] / data_1['height']
        ratio=np.array(ratio)

        position = multi_func(ratio, *tuple(fit_para))
        hist, bin_edges = np.histogram(position, bins=bin_num, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # ビンの中心を計算

        # 初期推定値の設定
        initial_guess = [np.max(hist), np.mean(position), np.std(position)]

        # ガウスフィッティング
        popt, pcov = curve_fit(gaussian, bin_centers, hist, p0=initial_guess, maxfev=10000)
        amp_fit, mean_fit, stddev_fit = popt

        # 半値全幅 (FWHM) の計算
        fwhm = 2 * stddev_fit * np.sqrt(2 * np.log(2))
        fwhms.append(fwhm)

        plt.hist(position, bins=bin_num, density=True, alpha=0.6, label=f"abs-{i}",color=cm.hsv(float(cnt) / float(len(para["position"]))))  # ヒストグラム
        cnt+=1
        x_fit = np.linspace(bin_edges[0], bin_edges[-1], 1000)  # フィッティング用のx
        plt.plot(x_fit, gaussian(x_fit, *popt), color="red")  # フィッティング曲線

    fwhms=np.array(fwhms)

    print(fwhms)

    np.savetxt(f"{output}/fwhms.dat", fwhms, fmt='%.5f', header='FWHM FWHM_Error')

    # グラフの設定
    
    plt.xlabel('Position[mm]',fontsize=14)
    plt.ylabel('Density',fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output}/Histgram.png",dpi=700)
    plt.show()

    plt.xlabel("Position[mm]")
    plt.ylabel("FWHM[mm]")
    plt.plot(para["position"],fwhms,marker="o")
    plt.show()

def Optimal_filter():
    with open(f"{output}/input.json", "r") as f:
        para = json.load(f)

    noise_spe_dens = np.loadtxt(f"{output}/noise_spectral_total_alpha71beta1.6.dat")
    noise_samples = len(noise_spe_dens)

    time = np.arange(0, 1 / para["rate"] * para["samples"], 1 / para["rate"])

    for j in [0,1]:
        for i in para["position"]:
            data_array=[]
            all_pulses=[]

            os.makedirs(f"{output}/{para["E"]}keV_{i}/pulse_optimal_filter",exist_ok=True)
                    
            for k in range(pulse_num):
                data = np.loadtxt(f"{output}/{para["E"]}keV_{i}/pulse_noise/CH{j}/CH{j}_{k}.dat")
                all_pulses.append(data)

            average_pulse=np.mean(all_pulses,axis=0)
            noise_spe=noise_spe_dens**2
            F = fft.fft(average_pulse)
            filt = fft.ifft(F[:noise_samples] / noise_spe[:noise_samples]).real
            filt=filt/np.max(filt)
            
            for pulse in all_pulses:
                pulse_f=pulse*filt
                peak,peak_av,peak_index = gp.peak(pulse_f,0,int(para['samples']),10,100)
                rise, rise_10, rise_90 = gp.risetime(pulse_f, peak_av, peak_index,0.9,0.1,para['rate'])

                column = [peak_av, rise]
                data_array.append(column)

            df_save = pd.DataFrame(data_array, columns=["height", "rise"],index=np.arange(0,pulse_num))
            df_save.to_csv(f"{output}/{para["E"]}keV_{i}/Pulse_optimal_filter/output_TES{j}.csv")

    plt_pulse=np.loadtxt(f"{output}/{para["E"]}keV_{para["position"][0]}/pulse_noise/CH0/CH0_0.dat")
    pulse_plt_f=plt_pulse*filt
    plt.plot(time,pulse_plt_f,label="Optimal filter")
    plt.plot(time,plt_pulse,label="normal")
    plt.xlabel("time[ms]")
    plt.grid()
    plt.show()

#MakePulse()
#FitRatios()
#MakeNoise()
#CheckPulse()
#MultiPulse()
MakeHistgram()