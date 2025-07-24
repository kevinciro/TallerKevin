import pandas as pd
import math

from t1dsim_ai.individual_model import DigitalTwin
import numpy as np
import matplotlib.pyplot as plt

from t1dsim_ai.utils.metrics import get_TAR180, get_TBR70, get_TIR


from t1dsim_ai.utils.metrics import get_TIR, get_TBR70, get_TAR180
""" from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error """

def mean_absolute_error(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("Las listas deben tener la misma longitud.")
    
    error_total = 0
    for real, pred in zip(y_true, y_pred):
        error_total += abs(real - pred)
    
    mae = error_total / len(y_true)
    return mae

def root_mean_squared_error(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("Las listas deben tener la misma longitud.")
    
    mse = 0
    for real, pred in zip(y_true, y_pred):
        mse += (real - pred) ** 2
    
    mse /= len(y_true)
    rmse = math.sqrt(mse)
    return rmse

def mean_absolute_percentage_error(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("Las listas deben tener la misma longitud.")
    
    error_total = 0
    for real, pred in zip(y_true, y_pred):
        if real == 0:
            raise ValueError("MAPE no está definido cuando y_true contiene ceros.")
        error_total += abs((real - pred) / real)
    
    mape = (error_total / len(y_true)) * 100  # En porcentaje
    return mape


df_simulation = pd.read_csv("./example_model/data_example.csv")
df_simulation = df_simulation[~df_simulation.is_train]
myDigitalTwin = DigitalTwin(n_digitalTwin=0, custom_DT="./example_model/DT_Example/")
df_simulation = myDigitalTwin.simulate(df_simulation.iloc[1 * 12 * 24 : 2 * 12 * 24])


print("cgm_Actual")
print("TIR: " + str(get_TIR(df_simulation["cgm_Actual"])))
print("TAR: " + str(get_TAR180(df_simulation["cgm_Actual"])))
print("TBR: " + str(get_TBR70(df_simulation["cgm_Actual"])))
print("---------------------------------")
print("cgm_NNPop")
print("RMSE: " + str(root_mean_squared_error(df_simulation["cgm_Actual"], df_simulation["cgm_NNPop"])))
print("ME: " + str(mean_absolute_error(df_simulation["cgm_Actual"], df_simulation["cgm_NNPop"])))
print("MARE: " + str(mean_absolute_percentage_error(df_simulation["cgm_Actual"], df_simulation["cgm_NNPop"])))
print("TIR: " + str(get_TIR(df_simulation["cgm_NNPop"])))
print("TAR: " + str(get_TAR180(df_simulation["cgm_NNPop"])))
print("TBR: " + str(get_TBR70(df_simulation["cgm_NNPop"])))
print("---------------------------------")
print("cgm_NNDT")
print("RMSE: " + str(root_mean_squared_error(df_simulation["cgm_Actual"], df_simulation["cgm_NNDT"])))
print("ME: " + str(mean_absolute_error(df_simulation["cgm_Actual"], df_simulation["cgm_NNDT"])))
print("MARE: " + str(mean_absolute_percentage_error(df_simulation["cgm_Actual"], df_simulation["cgm_NNDT"])))
print("TIR: " + str(get_TIR(df_simulation["cgm_NNDT"])))
print("TAR: " + str(get_TAR180(df_simulation["cgm_NNDT"])))
print("TBR: " + str(get_TBR70(df_simulation["cgm_NNDT"])))


time = np.arange(len(df_simulation))
sup_limit = np.ones(time.shape)*180
inf_limit = np.ones(time.shape)*70

fig, axs = plt.subplots(5, 2, figsize=(18, 9), sharex=True)

""" 
    Solo los valores que dependan de Q1 seran diferentes, ya que el modelo individual 
    solo calcula Q1. En este caso solo se veran afectados Q1 y Q2.
"""

# Q1
axs[0,0].plot(time, df_simulation["cgm_NNPop"], "r", label="Pop")
axs[0,0].plot(time, df_simulation["cgm_NNDT"], "g", label= "DT")
axs[0,0].plot(time, df_simulation["cgm_Actual"], "b", label="Real")
axs[0,0].plot(time, sup_limit, "k")
axs[0,0].plot(time, inf_limit, "k")
axs[0,0].grid()
axs[0,0].set_title("Q1")
axs[0,0].legend()

#Q2
axs[0,1].plot(time, df_simulation["state_Q2"], "r", label="Pop")
axs[0,1].plot(time, df_simulation["state_Q2_DT"], "g", label= "DT")
axs[0,1].grid()
axs[0,1].set_title("Q2")
axs[0,1].legend()

#S1
axs[1,0].plot(time, df_simulation["state_S1"], "r", label="Pop")
axs[1,0].plot(time, df_simulation["state_S1_DT"], "g", label= "DT")
axs[1,0].grid()
axs[1,0].set_title("S1")
axs[1,0].legend()

#S2
axs[1,1].plot(time, df_simulation["state_S2"], "r", label="Pop")
axs[1,1].plot(time, df_simulation["state_S2_DT"], "g", label= "DT")
axs[1,1].grid()
axs[1,1].set_title("S2")
axs[1,1].legend()

#I
axs[2,0].plot(time, df_simulation["state_I"], "r", label="Pop")
axs[2,0].plot(time, df_simulation["state_I_DT"], "g", label= "DT")
axs[2,0].grid()
axs[2,0].set_title("I")
axs[2,0].legend()

#X1
axs[2,1].plot(time, df_simulation["state_X1"], "r", label="Pop")
axs[2,1].plot(time, df_simulation["state_X1_DT"], "g", label= "DT")
axs[2,1].grid()
axs[2,1].set_title("X1")
axs[2,1].legend()

#X2
axs[3,0].plot(time, df_simulation["state_X2"], "r", label="Pop")
axs[3,0].plot(time, df_simulation["state_X2_DT"], "g", label= "DT")
axs[3,0].grid()
axs[3,0].set_title("X2")
axs[3,0].legend()

#X3
axs[3,1].plot(time, df_simulation["state_X3"], "r", label="Pop")
axs[3,1].plot(time, df_simulation["state_X3_DT"], "g", label= "DT")
axs[3,1].grid()
axs[3,1].set_title("X3")
axs[3,1].legend()

#Ug
axs[4,0].plot(time, df_simulation["state_Ug"], "r", label="Pop")
axs[4,0].plot(time, df_simulation["state_Ug_DT"], "g", label= "DT")
axs[4,0].grid()
axs[4,0].set_title("Ug")
axs[4,0].legend()

#Ug2
axs[4,1].plot(time, df_simulation["state_Ug2"], "r", label="Pop")
axs[4,1].plot(time, df_simulation["state_Ug2_DT"], "g", label= "DT")
axs[4,1].grid()
axs[4,1].set_title("Ug2")
axs[4,1].legend()

plt.show()