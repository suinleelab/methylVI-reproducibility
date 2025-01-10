import argparse
import os
import sys
import time

import mudata
import numpy as np
import scanpy as sc
import scanpy.external as sce
from methyl_vi.model import MethylVI
from scvi._settings import settings

import constants
from constants import batch_keys, label_keys
from seurat_class import SeuratIntegration

parser = argparse.ArgumentParser()
parser.add_argument(
    "dataset",
    type=str,
    choices=["Liu2021_mixed"],
    help="Which dataset to use for the experiment"
)
parser.add_argument(
    "method",
    type=str,
    choices=[
        "methylVI",
        "harmony",
        "scanorama",
        "mnn",
        "seurat",
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
    "--random_seed",
    type=int,
    default=123,
    help="Random seed to use for experiment"
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
    type=int,
    default=2500,
    help="How many highly variable features to use"
)

parser.add_argument(
    "--likelihood",
    type=str,
    choices=[
        "binomial",
        "betabinomial"
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

mdata_file = os.path.join(
    "/projects/leelab3/methylVI",
    args.dataset,
    "data",
    f"{args.var_dim}_{args.n_features}_features.h5mu",
)

mdata = mudata.read_h5mu(mdata_file)
print(f"Data read from {mdata_file}")

settings.seed = args.random_seed

if args.method == "methylVI":
    MethylVI.setup_mudata(
        mdata,
        mc_layer="mc",
        cov_layer="cov",
        batch_key=batch_keys[args.dataset],
        methylation_modalities={
            "mCG": "mCG",
            "mCH": "mCH"
        },
        covariate_modalities={
            "batch_key": "mCG"
        },
    )

    model = MethylVI(
        mdata,
        n_latent=args.n_latent,
        likelihood=args.likelihood,
        dispersion=args.dispersion
    )

    start = time.time()
    model.train(max_epochs=args.max_epochs, early_stopping=True)
    end = time.time()

    results_dir = os.path.join(
        constants.DEFAULT_RESULTS_PATH,
        args.dataset + "_integration",
        args.var_dim,
        args.method,
        f"likelihood_{args.likelihood}",
        f"dispersion_{args.dispersion}",
        f"latent_{args.n_latent}",
        str(args.random_seed),
    )

    os.makedirs(results_dir, exist_ok=True)

    model.save(
        os.path.join(results_dir, "model.ckpt"),
        overwrite=True
    )
    np.save(
        os.path.join(results_dir, "latent_representations.npy"),
        model.get_latent_representation()
    )

elif args.method == 'seurat':
    adata_full = mdata['mCG']

    sc.pp.pca(mdata['mCG'])
    sc.pp.pca(mdata['mCH'])
    adata_full.obsm['X_pca'] = np.concatenate([
        mdata['mCG'].obsm['X_pca'] / mdata['mCG'].obsm['X_pca'].std(),
        mdata['mCH'].obsm['X_pca'] / mdata['mCH'].obsm['X_pca'].std()
    ], axis=1)

    adatas = [
        adata_full[adata_full.obs[batch_keys[args.dataset]] == x] for x in adata_full.obs[batch_keys[args.dataset]].unique()
    ]

    min_sample = adata_full.obs[batch_keys[args.dataset]].value_counts().min()
    integrator = SeuratIntegration()

    anchor = integrator.find_anchor(
        adatas,
        k_local=None,
        key_local="X_pca",
        k_anchor=5,
        key_anchor="X",
        dim_red="cca",
        max_cc_cells=100000,
        k_score=30,
        k_filter=min(200, min_sample),
        scale1=True,
        scale2=True,
        n_components=20,
        n_features=200,
        alignments=[[[0], [1]]],
    )

    corrected = integrator.integrate(
        key_correct="X_pca",
        row_normalize=True,
        k_weight=100,
        sd=1,
        alignments=[[[0], [1]]],
    )

    for i, adata in enumerate(adatas):
        adata.obsm['X_pca_integrate'] = corrected[i]

    corrected_adata = sc.AnnData.concatenate(*adatas)
    corrected_adata.obs = corrected_adata.obs.reindex(
        [x[:-2] for x in corrected_adata.obs.index]
    )
    corrected_adata = corrected_adata[mdata['mCG'].obs.index, :]

    results_dir = os.path.join(
        constants.DEFAULT_RESULTS_PATH,
        args.dataset + "_integration",
        args.var_dim,
        args.method,
        str(args.random_seed),
    )

    os.makedirs(results_dir, exist_ok=True)

    np.save(
        os.path.join(results_dir, "latent_representations.npy"),
        corrected_adata.obsm['X_pca_integrate']
    )

elif args.method == "scanorama":
    adata_full = mdata['mCG']

    sc.pp.pca(mdata['mCG'])
    sc.pp.pca(mdata['mCH'])
    adata_full.obsm['X_pca'] = np.concatenate([
        mdata['mCG'].obsm['X_pca'] / mdata['mCG'].obsm['X_pca'].std(),
        mdata['mCH'].obsm['X_pca'] / mdata['mCH'].obsm['X_pca'].std()
    ], axis=1)

    adatas = [
        adata_full[adata_full.obs[batch_keys[args.dataset]] == x] for x in adata_full.obs[batch_keys[args.dataset]].unique()
    ]

    adata = sc.AnnData.concatenate(*adatas)
    sce.pp.scanorama_integrate(adata, key=batch_keys[args.dataset])
    adata.obs = adata.obs.reindex(
        [x[:-2] for x in adata.obs.index]
    )
    corrected_adata = adata[mdata['mCG'].obs.index, :]

    results_dir = os.path.join(
        constants.DEFAULT_RESULTS_PATH,
        args.dataset + "_integration",
        args.var_dim,
        args.method,
        str(args.random_seed),
    )

    os.makedirs(results_dir, exist_ok=True)

    np.save(
        os.path.join(results_dir, "latent_representations.npy"),
        corrected_adata.obsm['X_scanorama']
    )

elif args.method == "harmony":
    # Use this as a placeholder to store PCA results from both mCG and mCH
    adata_full = mdata['mCG']

    sc.pp.pca(mdata['mCG'])
    sc.pp.pca(mdata['mCH'])
    adata_full.obsm['X_pca'] = np.concatenate([
        mdata['mCG'].obsm['X_pca'] / mdata['mCG'].obsm['X_pca'].std(),
        mdata['mCH'].obsm['X_pca'] / mdata['mCH'].obsm['X_pca'].std()
    ], axis=1)
    sce.pp.harmony_integrate(adata_full, batch_keys[args.dataset], max_iter_harmony=100)

    results_dir = os.path.join(
        constants.DEFAULT_RESULTS_PATH,
        args.dataset + "_integration",
        args.var_dim,
        args.method,
        str(args.random_seed),
    )

    os.makedirs(results_dir, exist_ok=True)

    np.save(
        os.path.join(results_dir, "latent_representations.npy"),
        adata_full.obsm['X_pca_harmony']
    )

elif args.method == 'mnn':
    sc.pp.pca(mdata['mCG'])
    sc.pp.pca(mdata['mCH'])

    adata_full = sc.AnnData(
        X=np.concatenate([
            mdata['mCG'].obsm['X_pca'] / mdata['mCG'].obsm['X_pca'].std(),
            mdata['mCH'].obsm['X_pca'] / mdata['mCH'].obsm['X_pca'].std()
        ], axis=1),
        obs=mdata['mCG'].obs
    )

    adatas = [
        adata_full[adata_full.obs[batch_keys[args.dataset]] == x] for x in adata_full.obs[batch_keys[args.dataset]].unique()
    ]

    corrected_adata = sce.pp.mnn_correct(*adatas)[0] # This outputs a few things, the anndata object is the first item

    # MNN appends -0, -1, etc. to samples from the 0th, 1st, etc. batch.
    # Here we remove those tags
    corrected_adata.obs = corrected_adata.obs.reindex(
        [x[:-2] for x in corrected_adata.obs.index]
    )

    # Reorder to match the original indexing
    corrected_adata = corrected_adata[mdata['mCG'].obs.index, :]

    results_dir = os.path.join(
        constants.DEFAULT_RESULTS_PATH,
        args.dataset + "_integration",
        args.var_dim,
        args.method,
        str(args.random_seed),
    )

    os.makedirs(results_dir, exist_ok=True)

    np.save(
        os.path.join(results_dir, "latent_representations.npy"),
        corrected_adata.X
    )
