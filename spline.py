import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

def spline_model(x, a, b):
    return a * x + b

def FindPosition(Fit_para,ratios):
    print("A")
    B_right = Fit_para[:, 1]
    B_left = Fit_para[:, 0]

    result=[]

    

def Reso(Data_path):
    fit_para_path=f"{Data_path}/ratios.txt"
    fit_para=np.loadtxt(fit_para_path, delimiter=',')

    ratio_base = fit_para[:, 1]
    posi_base = fit_para[:, 0]

    spline = UnivariateSpline(ratio_base, posi_base, k=3, s=0)

    
