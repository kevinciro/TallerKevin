from t1dsim_ai.individual_model import DigitalTwin
from t1dsim_ai.create_scenarios import digitalTwin_scenario
import numpy as np
import matplotlib.pyplot as plt


myDigitalTwin = DigitalTwin(n_digitalTwin=1)

df_simulation = myDigitalTwin.simulate(
    digitalTwin_scenario(
        meal_size_array=[75, 90, 30],  # g
        meal_time_fromStart_array=[60, 60 * 4, 60 * 11],  # min from start of simulation
        init_cgm=80,  # mg/dL
        basal_insulin=1,  # U/h
        carb_ratio=12,
        sim_time=24 * 60,
        hr=100,  # int or array of len sim_time
        initial_time="08:00:00",
        exercise_time=60 * 2,
    )
)

# Visualization

color_AIDT = "peru"
color_AIPop = "#0072B2"

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

time = np.arange(len(df_simulation))

ax1.plot(time, df_simulation.cgm_NNPop, ".", ms=5, c=color_AIPop)
ax1.plot(time, df_simulation.cgm_NNDT, ".", ms=5, c=color_AIDT)

ax1.plot(-1, -1, "o", ms=5, c=color_AIPop, label="NN-based population")
ax1.plot(-1, -1, "o", ms=5, c=color_AIDT, label="NN-based digital twin")

ax1.axhspan(70, 180, facecolor="gray", alpha=0.1)

for y in [70, 180, 250]:
    ax1.axhline(y, color="k", alpha=0.2, lw=0.3)

ax1.legend(loc=9, ncol=3, frameon=False)
ax1.set_ylim(40, 380)
ax1.set_ylabel("CGM [mg/dL]")
for location in ["left", "right", "top", "bottom"]:
    ax1.spines[location].set_linewidth(0.1)

ax2.plot(time, df_simulation.input_insulin, c="k", lw=0.7)
ax2.set_ylabel("Insulin [U/h]")
for location in ["left", "right", "top", "bottom"]:
    ax2.spines[location].set_linewidth(0.1)
ax2.spines["top"].set_linewidth(0.5)
ax2_carbs = ax2.twinx()
color = "tab:red"
ax2_carbs.plot(
    df_simulation.loc[df_simulation.input_meal_carbs != 0].index,
    df_simulation.loc[df_simulation.input_meal_carbs != 0, "input_meal_carbs"],
    "o",
    color=color,
)
ax2_carbs.set_ylabel("Meal carbs [g]", color=color)
ax2_carbs.tick_params(axis="y", labelcolor=color)
ax2_carbs.spines["right"].set_position(("axes", 0))
for location in ["left", "right", "top", "bottom"]:
    ax2_carbs.spines[location].set_linewidth(0)
ax2_hr = ax2.twinx()
color = "tab:green"
ax2_hr.plot(time, df_simulation.heart_rate, lw=0.5, color=color)
ax2_hr.set_ylabel("Heart rate [BPM]", color=color)
ax2_hr.tick_params(axis="y", labelcolor=color)
for location in ["left", "right", "top", "bottom"]:
    ax2_hr.spines[location].set_linewidth(0)
ax2_pa = ax2.twinx()
color = "tab:purple"
ax2_pa.plot(time, df_simulation.sleep_efficiency, lw=1, color=color)
for location in ["left", "right", "top", "bottom"]:
    ax2_pa.spines[location].set_linewidth(0)

ax2_pa.set_ylabel("Sleep efficiency", color=color, labelpad=-20)
ax2_pa.tick_params(axis="y", labelcolor=color)
ax2_pa.set_yticks([0, 1], [0, 1])
ax2_pa.tick_params(axis="y", direction="in", pad=-15)


ax2.set_xticks(np.arange(0, 1 * 12 * 24 + 1, 12 * 12), np.arange(0, 1.1, 0.5))
ax2.set_xlabel("Simulation time [day]")
ax2.set_xlim(-12 * 3, 1 * 12 * 24 + 12 * 2)

plt.subplots_adjust(hspace=0)

plt.show()

# plt.savefig(
#    "img/example_digitaltwin" + str(myDigitalTwin.n_digitalTwin) + ".png",
#    dpi=500,
#    bbox_inches="tight",
# )
