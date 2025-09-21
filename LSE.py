import json
import ctypes
import numpy as np
from scipy.optimize import minimize
import pulse_model

output = "f:/hata/average_pulse/pulses"

eta = 101
amp = 10

def MakePulse(para):   
    ch0,ch1=pulse_model.model(para)

    pulses = []

    for i in range(len(para["position"])):
        pulses.appen(ch0[i])

    return pulses

def LoadExprimetalData(para):
    pulses=[]
    data = np.loadtxt(f"{output}/{para["E"]}keV_{para["position"][0]}/pulse/CH0/CH0_1.dat")
    return pulses



with open(f"{output}/input.json", "r") as f:
    para = json.load(f)

ExperimentalData=[]

cnt=1
for i in range(1,8):  
    data =  np.loadtxt(f"f:/hata/average_pulse/average_pulse-block_{cnt}.dat")/amp*eta
    data = data[5000:10000]
    ExperimentalData.append(data)
    cnt+=1

initial_params = [7.9e-10,
                7.9e-12,
                1.5e-07,
                8.2e-09,
                1.68e-08
                ]

def err_func(params):
    para["C_abs"] = params[0]
    para["C_tes"] = params[1]
    para["G_abs-abs"] = params[2]
    para["G_abs-tes"] = params[3]
    para["G_tes-bath"] = params[4]
    with open(f"{output}/input.json", "w") as f:
        json.dump(para, f)
    pulses = MakePulse(para)
    err = 0
    for i in range(len(ExperimentalData)):
        err += np.sum((pulses[i] - ExperimentalData[i])**2)

    return err

result = minimize(err_func, initial_params, method="Nelder-Mead", 
                  options={'maxiter': 500, 'disp': True, 'xatol': 1e-6, 'fatol': 1e-6})