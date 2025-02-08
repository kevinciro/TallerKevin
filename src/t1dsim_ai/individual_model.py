from t1dsim_ai.options import (
    n_neurons_pop,
    hidden_compartments,
    states,
    inputs,
    input_ind,
    idx_robust,
)
import torch
import torch.nn as nn
from pathlib import Path
import os
import numpy as np
from pickle import load
from t1dsim_ai.population_model import CGMOHSUSimStateSpaceModel_V2

from t1dsim_ai.utils.preprocess import scaler as scaler_pop
from t1dsim_ai.utils.preprocess import scaler_inverse, scale_single_state


class WeightClipper(object):
    def __init__(self, min=-1, max=1):

        self.min = min
        self.max = max

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, "weight"):
            w = module.weight.data
            w = w.clamp(self.min, self.max)


class ForwardEulerSimulator(nn.Module):

    """This class implements prediction/simulation methods for the SS models structure

     Attributes
     ----------
     ss_pop_model: nn.Module
                   The population-level neural state space models to be fitted
    ss_ind_model: nn.Module
                   The individual-level neural state space models to be fitted
     ts: float
         models sampling time

    """

    def __init__(self, ss_pop_model, ss_ind_model, path_scaler, ts=1.0):
        super(ForwardEulerSimulator, self).__init__()
        self.ss_pop_model = ss_pop_model
        self.ss_ind_model = ss_ind_model

        self.ts = ts
        self.cgm_min = scale_single_state(40, "Q1", path_scaler)
        self.cgm_max = scale_single_state(400, "Q1", path_scaler)

    def adjust_cgm(self, x):
        x[x > self.cgm_max] = self.cgm_max
        x[x < self.cgm_min] = self.cgm_min

        return x

    def forward(
        self, x0_batch: torch.Tensor, u_batch: torch.Tensor, u_batch_ind, is_pers=True
    ) -> torch.Tensor:
        """Multi-step simulation over (mini)batches

        Parameters
        ----------
        x0_batch: Tensor. Size: (q, n_x)
             Initial state for each subsequence in the minibatch

        u_batch: Tensor. Size: (m, q, n_u)
            Input sequence for each subsequence in the minibatch

        Returns
        -------
        Tensor. Size: (m, q, n_x)
            Simulated state for all subsequences in the minibatch

        """

        # X_sim_list: List[torch.Tensor] = []
        X_sim_list: [torch.Tensor] = []

        x_step = x0_batch

        for step in range(u_batch.shape[0]):
            u_step = u_batch[step]

            if is_pers:
                u_ind_step = u_batch_ind[step]

            X_sim_list += [x_step]

            dx_pop = self.ss_pop_model(x_step, u_step)

            dx = dx_pop
            if is_pers:
                dx_ind = self.ss_ind_model(x_step, u_step, u_ind_step)
                dx[:, 0] += dx_ind[:, 0]

            x_step = x_step + self.ts * dx

            if (len(x_step.shape)) == 1:
                x_step[0] = self.adjust_cgm(x_step[0])

            else:
                x_step[:, 0] = self.adjust_cgm(x_step[:, 0])

        X_sim = torch.stack(X_sim_list, 0)
        return X_sim


class CGMIndividual(nn.Module):
    def __init__(self, hidden_compartments, init_small=True):
        super(CGMIndividual, self).__init__()

        # NN - Model
        layers_model = []
        for i in range(len(hidden_compartments["models"]) - 2):
            layers_model.append(
                nn.Linear(
                    hidden_compartments["models"][i],
                    hidden_compartments["models"][i + 1],
                )
            )
            layers_model.append(nn.ReLU())

        layers_model.append(nn.Linear(hidden_compartments["models"][-2], 1))
        self.net_model = nn.Sequential(*layers_model)

        clipper = WeightClipper()

        if init_small:
            networks = {"Model": self.net_model}

            for key in networks.keys():
                net = networks[key]
                for m in net.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, mean=0, std=1e-4)
                        nn.init.constant_(m.bias, val=0)

                net.apply(clipper)

    def forward(self, in_x, u_pop, u_ind):
        # q1, q2, s1, s2, I, x1, x2, x3, c2, c1
        q1, q2, _, _, _, x1, _, x3, c2, _ = (
            in_x[..., [0]],
            in_x[..., [1]],
            in_x[..., [2]],
            in_x[..., [3]],
            in_x[..., [4]],
            in_x[..., [5]],
            in_x[..., [6]],
            in_x[..., [7]],
            in_x[..., [8]],
            in_x[..., [9]],
        )

        inp = torch.cat((q1, q2, x1, x3, c2, u_ind), -1)
        dQ1_Ind = self.net_model(inp)

        return dQ1_Ind


class DigitalTwin:
    def __init__(self, n_digitalTwin=0, device=torch.device("cpu"), ts=5):
        self.ts = ts
        self.device = device

        self.n_digitalTwin = n_digitalTwin
        digitalTwin_list = [
            f.path
            for f in os.scandir(Path(__file__).parent / "models/IndividualModel/")
            if f.is_dir()
        ]
        digitalTwin_list.sort()
        self.digital_twin_folder = digitalTwin_list[self.n_digitalTwin]

        self.setup_simulator()

    def setup_simulator(self):
        # Population Model
        ss_pop_model = CGMOHSUSimStateSpaceModel_V2(n_feat=n_neurons_pop)
        ss_pop_model.to(self.device)
        ss_pop_model.load_state_dict(
            torch.load(
                Path(__file__).parent
                / "models/PopulationModel/population_model_05022024_epoch_15.pt"
            )
        )

        # Individual Model
        ss_individual_model = CGMIndividual(hidden_compartments=hidden_compartments)
        ss_individual_model.to(self.device)
        ss_individual_model.load_state_dict(
            torch.load(self.digital_twin_folder + "/individual_model.pt")
        )

        for name, param in ss_pop_model.named_parameters():
            param.requires_grad = False
        for name, param in ss_individual_model.named_parameters():
            param.requires_grad = False

        # Simulator
        self.nn_solution = ForwardEulerSimulator(
            ss_pop_model,
            ss_individual_model,
            str(Path(__file__).parent) + "/models/PopulationModel/",
            ts=self.ts,
        )

        self.scaler_featsRobust = load(
            open(self.digital_twin_folder + "/scaler_robust.pkl", "rb")
        )

    def prepare_data(self, df_scenario):

        sim_time = len(df_scenario)
        batch_start = np.array([0], dtype=np.int)
        batch_idx = batch_start[:, np.newaxis] + np.arange(sim_time)

        x_est = np.array(df_scenario[states].values).astype(np.float32)
        u_pop = np.array(df_scenario[inputs].values).astype(np.float32)
        u_ind = np.array(df_scenario[input_ind].values).astype(np.float32)

        # Scale states and inputs from the population models
        x_est, u_pop = scaler_pop(
            x_est, u_pop, str(Path(__file__).parent) + "/models/PopulationModel/", False
        )

        # Scale new inputs
        u_ind[:, idx_robust] = self.scaler_featsRobust.transform(u_ind[:, idx_robust])

        u_pop = u_pop.reshape(-1, sim_time, len(inputs))[0, batch_idx, :]
        u_ind = u_ind.reshape(-1, sim_time, len(input_ind))[0, batch_idx, :]

        x0_est = torch.tensor(
            df_scenario.loc[0, states].astype(float).values.reshape(1, -1),
            dtype=torch.float32,
        ).to(self.device)

        u_pop = torch.tensor(u_pop[[0], batch_idx.T], dtype=torch.float32).to(
            self.device
        )
        u_ind = torch.tensor(u_ind[[0], batch_idx.T], dtype=torch.float32).to(
            self.device
        )

        return x0_est, u_pop[:, [0], :], u_ind[:, [0], :]

    def simulate(self, df_scenario_original):
        # Prepare data
        df_scenario = df_scenario_original.copy()
        x0_est, u_pop, u_ind = self.prepare_data(df_scenario)

        with torch.no_grad():
            x_sim_pop = self.nn_solution(x0_est, u_pop, None, is_pers=False)
            df_scenario[states] = scaler_inverse(
                x_sim_pop[:, 0, :].to("cpu").detach().numpy(),
                str(Path(__file__).parent) + "/models/PopulationModel/",
            )

            x_sim_DT = self.nn_solution(x0_est, u_pop, u_ind, is_pers=True)

            df_scenario[[s + "_DT" for s in states]] = scaler_inverse(
                x_sim_DT[:, 0, :].to("cpu").detach().numpy(),
                str(Path(__file__).parent) + "/models/PopulationModel/",
            )

        df_scenario["cgm_NNPop"] = df_scenario["output_cgm"]
        df_scenario["cgm_NNDT"] = df_scenario["output_cgm_DT"]

        return df_scenario
