import torch.nn as nn
from pickle import load
import glob
from t1dsim_ai.Population.population_model import CGMOHSUSimStateSpaceModel_V2

from t1dsim_ai.utils.utils import *
from t1dsim_ai.utils.preprocess import scaler as scaler_pop
from t1dsim_ai.utils.preprocess import scale_single_state


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

    """This class implements prediction/simulation methods for the SS model structure

     Attributes
     ----------
     ss_pop_model: nn.Module
                   The population-level neural state space model to be fitted
    ss_ind_model: nn.Module
                   The individual-level neural state space model to be fitted
     ts: float
         model sampling time

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

        X_sim_list: List[torch.Tensor] = []
        # PA_sim_list: List[torch.Tensor] = []

        x_step = x0_batch
        # pa=torch.zeros((x0_batch.shape[0],1))

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
        # PA_sim = torch.stack(PA_sim_list, 0)
        return X_sim  # ,PA_sim


class CGMIndividual(nn.Module):
    def __init__(self, hidden_compartments, init_small=True):
        super(CGMIndividual, self).__init__()

        # PA - Compartment
        # self.net_dPA = nn.Sequential(
        #    nn.Linear(2, 32),
        #    nn.ReLU(),
        #    nn.Linear(32, 1),
        # )

        # NN - Model
        layers_model = []
        for i in range(len(hidden_compartments["model"]) - 2):
            layers_model.append(
                nn.Linear(
                    hidden_compartments["model"][i], hidden_compartments["model"][i + 1]
                )
            )
            layers_model.append(nn.ReLU())

        layers_model.append(nn.Linear(hidden_compartments["model"][-2], 1))
        self.net_model = nn.Sequential(*layers_model)

        clipper = WeightClipper()

        if init_small:
            networks = {"Model": self.net_model}  # , 'PA': self.net_dPA}

            for key in networks.keys():
                net = networks[key]
                for m in net.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, mean=0, std=1e-4)
                        nn.init.constant_(m.bias, val=0)

                net.apply(clipper)

    def forward(self, in_x, u_pop, u_ind):
        q1, q2, s1, s2, I, x1, x2, x3, c2, c1 = (
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

        # hr = u_ind[...,[0]]
        # NN1(hr,pa)
        # in_1 = torch.cat((hr,pa), -1)  # concatenate
        # dPA = self.net_dPA(in_1)

        # NN2(Q1,Q2,X1,X3,C2,PA,uInd)
        inp = torch.cat((q1, q2, x1, x3, c2, u_ind), -1)
        dQ1_Ind = self.net_model(inp)

        return dQ1_Ind  # , dPA


class DigitalTwin:
    def __init__(
        self,
        subjectID,
        df_subj,
        states,
        inputs_pop,
        input_ind,
        popModel,
        pathModel,
        device="cpu",
    ):

        self.subjectID = subjectID
        self.df_subj = df_subj
        self.device = device

        self.popModel = popModel
        self.pathModel = pathModel

        self.preprocess(states, inputs_pop, input_ind)

    def preprocess(self, states, inputs_pop, input_ind):

        self.scaler_featsRobust = load(
            open(self.pathModel + "/scaler_robust.pkl", "rb")
        )

        df_subj_test = self.df_subj

        sim_time_test = len(df_subj_test)
        batch_start = np.array([0], dtype=np.int)
        batch_idx = batch_start[:, np.newaxis] + np.arange(sim_time_test)

        x_est_test = np.array(df_subj_test[states].values).astype(np.float32)
        u_pop_test = np.array(df_subj_test[inputs_pop].values).astype(np.float32)
        u_ind_test = np.array(df_subj_test[input_ind].values).astype(np.float32)

        # Scale states and inputs from the population model
        x_est_test, u_pop_test = scaler_pop(
            x_est_test, u_pop_test, self.popModel, False
        )

        # Scale new inputs
        u_ind_test[:, idx_robust] = self.scaler_featsRobust.transform(
            u_ind_test[:, idx_robust]
        )

        self.u_pop_test = u_pop_test.reshape(-1, sim_time_test, len(inputs_pop))[
            0, batch_idx, :
        ]
        self.u_ind_test = u_ind_test.reshape(-1, sim_time_test, len(input_ind))[
            0, batch_idx, :
        ]

        self.u_pop = torch.tensor(
            self.u_pop_test[[0], batch_idx.T], dtype=torch.float32
        ).to(self.device)
        self.u_ind = torch.tensor(
            self.u_ind_test[[0], batch_idx.T], dtype=torch.float32
        ).to(self.device)

    def setup_simulator(self, n_neurons_pop, hidden_compartments, ts=5):

        # Load individual model
        self.individual_model = CGMIndividual(hidden_compartments=hidden_compartments)
        self.individual_model.to(self.device)
        self.individual_model.load_state_dict(
            torch.load(self.pathModel + "individual_model.pt")
        )

        # Load population model
        self.ss_pop_model = CGMOHSUSimStateSpaceModel_V2(n_feat=n_neurons_pop)
        self.ss_pop_model.to(self.device)
        self.ss_pop_model.load_state_dict(
            torch.load(glob.glob(self.popModel + "*pt")[0])
        )

        for name, param in self.ss_pop_model.named_parameters():
            param.requires_grad = False
        for name, param in self.individual_model.named_parameters():
            param.requires_grad = False

        # Simulator
        self.nn_solution = ForwardEulerSimulator(
            self.ss_pop_model, self.individual_model, self.popModel, ts=ts
        )
