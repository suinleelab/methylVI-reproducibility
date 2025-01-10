from tqdm import tqdm
import os
import shutil
import pandas as pd
from mudata import MuData
import mudata
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import globalenv
from rpy2.robjects import pandas2ri
import time
import constants
import pickle

pandas2ri.activate()

num_features = 2500

mdata_single_cell = mudata.read_h5mu(
    f"/projects/leelab2/metVI/Luo2017_Mouse/data/gene_{num_features}_features.h5mu"
)


excitatory_subtypes = [
    "mL2/3",
    "mL4",
    "mL5-1",
    "mDL-1",
    "mDL-2",
    "mDL-3",
    "mIn-1",
    "mL6-1",
    "mL6-2",
    "mL5-2"
]

def get_neuron_type(x):
    if x in excitatory_subtypes:
        return "Excitatory"
    elif x == "mVip":
        return "Inhibitory"
    elif x == "mPv":
        return "Inhibitory"
    else:
        return "Other"


def adata_to_scmet(adata):
    # scMET requires that all features have > 3 cells
    adata = adata[:, np.sum(adata.layers['cov'] > 0, axis=0) > 3]

    total_reads_df = pd.DataFrame(
        data=adata.layers['cov'],
        index=adata.obs.index,
        columns=adata.var.index
    )

    met_reads_df = pd.DataFrame(
        data=adata.layers['mc'],
        index=adata.obs.index,
        columns=adata.var.index
    )

    total_reads_df = total_reads_df.stack().reset_index()
    total_reads_df = total_reads_df.rename(
        columns={
            "cell": "Cell", "level_1": "Feature", 0: "total_reads"
        }
    )

    met_reads_df = met_reads_df.stack().reset_index()
    met_reads_df = met_reads_df.rename(
        columns={
            "cell": "Cell", "level_1": "Feature", 0: "met_reads"
        }
    )

    final_df = total_reads_df[["Feature", "Cell", "total_reads"]]
    final_df["met_reads"] = met_reads_df["met_reads"]

    final_df["met_reads"] = final_df["met_reads"].astype(np.int32)
    final_df["total_reads"] = final_df["total_reads"].astype(np.int32)

    final_df = final_df[final_df["total_reads"] > 0]

    return final_df

for seed in constants.DEFAULT_SEEDS:
    start = time.time()
    for modality in ['mCG', 'mCH']:
        adata = mdata_single_cell[modality]

        adata.obs['Coarse_type'] = [get_neuron_type(x) for x in adata.obs["Neuron type"]]

        excitatory_adata = adata[adata.obs['Coarse_type'] == "Excitatory"].copy()
        inhibitory_adata = adata[adata.obs['Coarse_type'] == "Inhibitory"].copy()

        # scMET errors out if any features are measured in < 3 cells
        excitatory_adata = excitatory_adata[:, np.sum(excitatory_adata.layers['cov'] > 0, axis=0) > 3]
        inhibitory_adata = inhibitory_adata[:, np.sum(inhibitory_adata.layers['cov'] > 0, axis=0) > 3]

        if modality == 'mCG':
            inhibitory_adata = inhibitory_adata[:, np.sum(inhibitory_adata.layers['cov'] > 0, axis=0) > 20]
            excitatory_adata = excitatory_adata[:, np.sum(excitatory_adata.layers['cov'] > 0, axis=0) > 20]

        excitatory_scmet = adata_to_scmet(excitatory_adata)
        inhibitory_scmet = adata_to_scmet(inhibitory_adata)

        ro.r["library"]("data.table")
        ro.r["library"]("scMET")

        globalenv['excitatory_df'] = excitatory_scmet

        t0 = time.time()
        ro.r("setDT(excitatory_df)")
        ro.r("excitatory_obj <- scmet(Y = excitatory_df, L = 4, iter = 10000, seed = 12)")
        t1 = time.time()
        excitatory_df_time = t1 - t0

        globalenv['inhibitory_df'] = inhibitory_scmet

        t0 = time.time()
        ro.r("setDT(inhibitory_df)")
        ro.r("inhibitory_obj <- scmet(Y = inhibitory_df, L = 4, iter = 10000, seed = 12)")
        t1 = time.time()
        inhibitory_df_time = t1 - t0

        ro.r(f'set.seed({seed})')

        ro.r(
        '''
            excitatory_vs_inhibitory_diff_obj <- scmet_differential(
                obj_A = excitatory_obj,
                obj_B = inhibitory_obj,
                evidence_thresh_m = 0.65,
                evidence_thresh_e = 0.65,
                group_label_A = "A",
                group_label_B = "B"
            )
        '''
        )

        ro.r(f'write.csv(excitatory_vs_inhibitory_diff_obj$diff_mu_summary, "Excitatory_vs_Inhibitory_{num_features}_{modality}_scmet_{seed}.csv")')

    end = time.time()
    training_time = end - start
    pickle.dump(training_time, open(f"scMET_{seed}_runtime.pkl", mode="wb"))
