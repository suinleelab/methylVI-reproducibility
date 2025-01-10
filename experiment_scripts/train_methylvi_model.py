import argparse
import os
import pickle
import sys
import time

import mudata
import numpy as np
from methyl_vi.model import MethylVI
from scvi._settings import settings

import constants

parser = argparse.ArgumentParser()
parser.add_argument(
    "dataset",
    type=str,
    choices=constants.DATASET_LIST,
    help="Which dataset to use for the experiment"
)
parser.add_argument(
    "method",
    type=str,
    choices=[
        "methylVI"
    ],
    help="Which model to train"
)

parser.add_argument(
    "--n_latent",
    type=int,
    default=20,
    help="Size of the model's latent space"
)

parser.add_argument(
    "--random_seeds",
    nargs="+",
    type=int,
    default=constants.DEFAULT_SEEDS,
    help="List of random seeds to use for experiments, with one model trained per "
    "seed.",
)

parser.add_argument(
    "--var_dim",
    type=str,
    choices=[
        "gene"
    ],
    default="gene",
    help="Which feature set to use (e.g. 100kbp windows or gene bodies)"
)

parser.add_argument(
    "--n_features",
    default=2500,
    help="How many highly variable features to use"
)

parser.add_argument(
    "--likelihood",
    type=str,
    choices=[
        "binomial",
        "betabinomial",
    ],
    default="betabinomial",
    help="Which distribution to use for methylVI's likelihood function"
)

parser.add_argument(
    "--max_epochs",
    type=int,
    default=500,
    help="Maximum number of epochs for model training"
)

parser.add_argument(
    "--dispersion",
    type=str,
    choices=[
        "gene",
        "gene-cell"
    ],
    default="gene",
    help="Whether to use a gene-specific dispersion parameter (shared across cells)"
         "or use different dispersion parameters for each gene for each cell."
)

args = parser.parse_args()
print(f"Running {sys.argv[0]} with arguments")
for arg in vars(args):
    print(f"\t{arg}={getattr(args, arg)}")


leelab3_datasets = [
    "Liu2021_snm3C-seq",
    "Liu2021_snmC-seq2",
    "Luo2022_snmCAT-seq",
    "Liu2021_mixed",
    "Luo2022_atlas",
]
if args.dataset in leelab3_datasets:
    dir = "/projects/leelab3/methylVI"
elif args.dataset == "Chien2023":
    dir = "/projects/leelab/methylVI"
else:
    dir = "/projects/leelab2/metVI"

if args.dataset == "Liu2021_snmC-seq2":
    mdata_file = os.path.join(
        dir,
        "Liu2021_mixed",
        "data",
        f"{args.var_dim}_{args.n_features}_features.h5mu",
    )
    mdata = mudata.read_h5mu(mdata_file)
    mdata = mdata[mdata.obs['Platform'] == 'snmcseq-2'].copy()
elif args.dataset == "Liu2021_snm3C-seq":
    mdata_file = os.path.join(
        dir,
        "Liu2021_mixed",
        "data",
        f"{args.var_dim}_{args.n_features}_features.h5mu",
    )
    mdata = mudata.read_h5mu(mdata_file)
    mdata = mdata[mdata.obs['Platform'] == 'snm-3C-seq'].copy()
elif args.dataset == "Luo2022_snmCAT-seq":
    mdata_file = os.path.join(
        dir,
        "Luo2022_atlas",
        "data",
        f"{args.var_dim}_{args.n_features}_features.h5mu",
    )
    mdata = mudata.read_h5mu(mdata_file)
    mdata = mdata[(mdata.obs['mCG:sample'] == "mCTseq_hs_21yr") | (mdata.obs['mCG:sample'] == "mCTseq_hs_29yr")].copy()
else:
    mdata_file = os.path.join(
        dir,
        args.dataset,
        "data",
        f"{args.var_dim}_{args.n_features}_features.h5mu",
    )
    mdata = mudata.read_h5mu(mdata_file)



print(f"Data read from {mdata_file}")

for seed in args.random_seeds:
    settings.seed = seed

    if args.method == "methylVI":
        MethylVI.setup_mudata(
            mdata,
            mc_layer="mc",
            cov_layer="cov",
            methylation_modalities={
                "mCG": "mCG",
                "mCH": "mCH"
            }
        )

        model = MethylVI(
            mdata,
            n_latent=args.n_latent,
            likelihood=args.likelihood,
            dispersion=args.dispersion
        )

        results_dir = os.path.join(
            constants.DEFAULT_RESULTS_PATH,
            args.dataset,
            args.var_dim,
            f"{args.n_features}_features",
            args.method,
            f"likelihood_{args.likelihood}",
            f"dispersion_{args.dispersion}",
            f"latent_{args.n_latent}",
            str(seed),
        )

        os.makedirs(results_dir, exist_ok=True)

        start = time.time()
        model.train(max_epochs=args.max_epochs, early_stopping=True)
        end = time.time()

        training_time = end - start
        pickle.dump(training_time, open(os.path.join(results_dir, "runtime.pkl"), mode="wb"))

        model.save(
            os.path.join(results_dir, "model.ckpt"),
            overwrite=True
        )
        np.save(
            os.path.join(results_dir, "latent_representations.npy"),
            model.get_latent_representation()
        )

        normalized_expression = model.get_normalized_expression(n_samples=10, return_numpy=True)
        pickle.dump(normalized_expression, open(os.path.join(results_dir, "normalized_expression.pkl"), mode="wb"))
