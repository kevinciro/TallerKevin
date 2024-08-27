
from t1dsim_ai.utils.preprocess import scale_inverse_Q1
from t1dsim_ai.Individual.individual_model import DigitalTwin
from t1dsim_ai.utils.utils import *
from t1dsim_ai.utils.utils_simulationsAI import getInitSSFromFile

import pandas as pd
import matplotlib.pyplot as plt
population_model_path = '../model/PopulationModel/'
individual_model_path = '../model/IndividualModel/Model0826/'
fileInitStates = '../model/initSteadyStates.csv'

df_scenario_subj = pd.read_csv('example_subjectX.csv')

myDigitalTwin = DigitalTwin('Subject_X',df_scenario_subj, states,inputs_pop,input_ind,population_model_path,individual_model_path)
myDigitalTwin.setup_simulator(n_neurons_pop,hidden_compartments)

initCGM = df_scenario_subj.cgm.values[0]
with torch.no_grad():
    x_sim_pop = myDigitalTwin.nn_solution(getInitSSFromFile(initCGM,fileInitStates).reshape(1,-1), myDigitalTwin.u_pop[:,[0],:], None,is_pers=False)
    x_sim_ind = myDigitalTwin.nn_solution(getInitSSFromFile(initCGM,fileInitStates).reshape(1,-1), myDigitalTwin.u_pop[:,[0],:], myDigitalTwin.u_ind[:,[0],:],is_pers=True)

df_scenario_subj['cgm_AIPop'] = scale_inverse_Q1(x_sim_pop[:, 0, [0]],population_model_path)
df_scenario_subj['cgm_AIDT'] = scale_inverse_Q1(x_sim_ind[:, 0, [0]],population_model_path)

# Visualization

color_AIDT = 'peru'
color_AIPop = '#0072B2'

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14,7), sharex=True)

time = np.arange(len(df_scenario_subj))

ax1.plot(time,df_scenario_subj.cgm,'k',label='Actual CGM')
ax1.plot(time,df_scenario_subj.cgm_AIPop,'.',ms=5,c = color_AIPop)
ax1.plot(time,df_scenario_subj.cgm_AIDT,'.',ms=5,c = color_AIDT)

ax1.plot(-1,-1,'o',ms=5,c = color_AIPop,label='AI-based population')
ax1.plot(-1,-1,'o',ms=5,c = color_AIDT,label='AI-based digital twin')

ax1.axhspan(70, 180, facecolor ='gray', alpha = 0.1)

for y in [70,180,250]:
    ax1.axhline(y, color='k',alpha=0.2,lw=0.3)

ax1.legend(loc=9,ncol=3, frameon=False)
ax1.set_ylim(40,380)
ax1.set_ylabel('CGM [mg/dL]')
for location in ['left', 'right', 'top', 'bottom']:
    ax1.spines[location].set_linewidth(0.1)

ax2.plot(time,df_scenario_subj.input_insulin_ODEPop,c='k',lw=0.7)
ax2.set_ylabel('Insulin [U/h]')
for location in ['left', 'right', 'top', 'bottom']:
    ax2.spines[location].set_linewidth(0.1)
ax2.spines['top'].set_linewidth(0.5)
ax2_carbs = ax2.twinx()
color ='tab:red'
ax2_carbs.plot(df_scenario_subj.loc[df_scenario_subj.input_meal_carbs_ODEPop!=0].index,df_scenario_subj.loc[df_scenario_subj.input_meal_carbs_ODEPop!=0,'input_meal_carbs_ODEPop'],'o',color=color)
ax2_carbs.set_ylabel('Meal carbs [g]', color=color)
ax2_carbs.tick_params(axis='y', labelcolor=color)
ax2_carbs.spines['right'].set_position(('axes', 0))
for location in ['left', 'right', 'top', 'bottom']:
    ax2_carbs.spines[location].set_linewidth(0)
ax2_hr = ax2.twinx()
color ='tab:green'
ax2_hr.plot(time,df_scenario_subj.heart_rate_WRTbaseline,lw=0.5,color=color)
#ax2_hr.plot(time,df_scenario_subj.change_heart_rate_WRTbaseline,lw=0.5,ls='--',color=color)
ax2_hr.set_ylabel('Heart rate [BPM]', color=color)
ax2_hr.tick_params(axis='y', labelcolor=color)
for location in ['left', 'right', 'top', 'bottom']:
    ax2_hr.spines[location].set_linewidth(0)
ax2_pa = ax2.twinx()
color ='tab:purple'
ax2_pa.plot(time,df_scenario_subj.sleep_efficiency,lw=1,color=color)
for location in ['left', 'right', 'top', 'bottom']:
    ax2_pa.spines[location].set_linewidth(0)

ax2_pa.set_ylabel('Sleep efficiency', color=color,labelpad=-20)
ax2_pa.tick_params(axis='y', labelcolor=color)
ax2_pa.set_yticks([0,1],[0,1])
ax2_pa.tick_params(axis="y",direction="in", pad=-15)


ax2.set_xticks(np.arange(0,1*12*24+1,12*12),np.arange(0,1.1,0.5))
ax2.set_xlabel('Simulation time [day]')
ax2.set_xlim(-12*3,1*12*24+12*2)

plt.subplots_adjust(hspace=0)

plt.savefig('img/figure_exampleSubject_X.png',dpi=500, bbox_inches='tight')
