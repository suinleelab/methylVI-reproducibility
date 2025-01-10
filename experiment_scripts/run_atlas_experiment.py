import argparse
import os
import sys

import anndata as ad
import mudata
import numpy as np
import scanpy as sc
import scanpy.external as sce
from methyl_vi.model import MethylVI, MethylANVI
from scvi._settings import settings
from sklearn.preprocessing import normalize

import constants
from constants import batch_keys, label_keys
from seurat_class import SeuratIntegration

parser = argparse.ArgumentParser()
parser.add_argument(
    "method",
    type=str,
    choices=[
        "methylVI_denovo",
        "methylANVI_arches",
        "methylANVI_denovo",
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

dataset = "Luo2022_atlas"
mdata_file = os.path.join(
    "/projects/leelab3/methylVI",
    dataset,
    "data",
    f"{args.var_dim}_{args.n_features}_features.h5mu",
)

mdata = mudata.read_h5mu(mdata_file)
print(f"Data read from {mdata_file}")

settings.seed = args.random_seed

excluded_sample = "NDARKD326LNK_hs_25yr"
batch_key = "sample"

if args.method == "methylANVI_arches":
    mdata.obs[label_keys[dataset]] = mdata['mCG'].obs[label_keys[dataset]]
    mdata_ref = mdata[mdata.obs[f'mCH:{batch_key}'] != excluded_sample].copy()
    mdata_query = mdata[mdata.obs[f'mCH:{batch_key}'] == excluded_sample].copy()
    mdata_query.obs[label_keys[dataset]] = "Unknown"

    mdata_ref.obs['dataset'] = "Reference"
    mdata_query.obs['dataset'] = "Query"

    MethylANVI.setup_anndata(
        mdata_ref,
        mc_layer="mc",
        cov_layer="cov",
        batch_key=batch_keys[dataset],
        labels_key=f"{label_keys[dataset]}",
        unlabeled_category="Unknown",
        methylation_modalities={
            "mCG": "mCG",
            "mCH": "mCH"
        },
        covariate_modalities={
            "batch_key": "mCG"
        },
    )

    model = MethylANVI(
        mdata_ref,
    )

    model.train(max_epochs=500, early_stopping=True)

    model_q = MethylANVI.load_query_mdata(
        mdata_query,
        model,
    )

    model_q.train(
        max_epochs=150,
        plan_kwargs=dict(weight_decay=0.0),
        early_stopping=True
    )

    # Hack to work around not being able to concatenate MuDatas
    adata_ref = mdata_ref['mCG']
    adata_query = mdata_query['mCG']

    adata_ref.obsm['vae'] = model.get_latent_representation()
    adata_query.obsm["vae"] = model_q.get_latent_representation(mdata_query)

    adata_full = ad.concat([adata_ref, adata_query])
    adata_full = adata_full[mdata['mCG'].obs.index, :]

    results_dir = os.path.join(
        constants.DEFAULT_RESULTS_PATH,
        dataset + "_integration",
        args.query_sample,
        args.var_dim,
        args.method,
        f"likelihood_{args.likelihood}",
        f"dispersion_{args.dispersion}",
        f"latent_{args.n_latent}",
        str(args.random_seed),
    )

    os.makedirs(results_dir, exist_ok=True)

    np.save(
        os.path.join(results_dir, "latent_representations.npy"),
        adata_full.obsm['vae']
    )

elif args.method == "methylVI_denovo":
    MethylVI.setup_mudata(
        mdata,
        mc_layer="mc",
        cov_layer="cov",
        batch_key=batch_key,
        unlabeled_category="Unknown",
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

    model.train(max_epochs=args.max_epochs, early_stopping=True)

    results_dir = os.path.join(
        constants.DEFAULT_RESULTS_PATH,
        dataset + "_integration",
        args.query_sample,
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

elif args.method == "methylANVI_denovo":
    mdata.obs[label_keys[dataset]] = mdata['mCG'].obs[label_keys[dataset]]
    mdata[mdata.obs[f'mCH:{batch_key}'] == excluded_sample].obs[label_keys[dataset]] = 'Unknown'

    MethylANVI.setup_anndata(
        mdata,
        mc_layer="mc",
        cov_layer="cov",
        batch_key=batch_keys[dataset],
        labels_key=f"{label_keys[dataset]}",
        unlabeled_category="Unknown",
        methylation_modalities={
            "mCG": "mCG",
            "mCH": "mCH"
        },
        covariate_modalities={
            "batch_key": "mCG"
        },
    )

    model = MethylANVI(
        mdata,
        n_latent=args.n_latent,
        likelihood=args.likelihood,
        dispersion=args.dispersion
    )

    model.train(max_epochs=args.max_epochs, early_stopping=True)

    results_dir = os.path.join(
        constants.DEFAULT_RESULTS_PATH,
        dataset + "_integration",
        args.query_sample,
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


elif args.method == "scanorama":
    adata_full = mdata['mCG']

    sc.pp.pca(mdata['mCG'])
    sc.pp.pca(mdata['mCH'])
    adata_full.obsm['X_pca'] = np.concatenate([
        mdata['mCG'].obsm['X_pca'] / mdata['mCG'].obsm['X_pca'].std(),
        mdata['mCH'].obsm['X_pca'] / mdata['mCH'].obsm['X_pca'].std()
    ], axis=1)

    adatas = [
        adata_full[adata_full.obs[batch_key] == x] for x in adata_full.obs[batch_key].unique()
    ]

    adata = sc.AnnData.concatenate(*adatas)
    sce.pp.scanorama_integrate(adata, key=batch_key)
    adata.obs = adata.obs.reindex(
        [x[:-2] for x in adata.obs.index]
    )
    corrected_adata = adata[mdata['mCG'].obs.index, :]

    results_dir = os.path.join(
        constants.DEFAULT_RESULTS_PATH,
        dataset + "_integration",
        args.query_sample,
        args.var_dim,
        args.method,
        str(args.random_seed),
    )

    os.makedirs(results_dir, exist_ok=True)

    np.save(
        os.path.join(results_dir, "latent_representations.npy"),
        adata_full.obsm['X_scanorama']
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
    sce.pp.harmony_integrate(adata_full, batch_key, max_iter_harmony=100)

    results_dir = os.path.join(
        constants.DEFAULT_RESULTS_PATH,
        dataset + "_integration",
        args.query_sample,
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

    corrected_adata = sce.pp.mnn_correct(
        *[adata_full[adata_full.obs[batch_key] == x] for x in adata_full.obs[batch_key].unique()],
    )[0]  # This outputs a few things, the anndata object is the first item

    # MNN appends -0, -1, etc. to samples from the 0th, 1st, etc. batch.
    # Here we remove those tags
    corrected_adata.obs = corrected_adata.obs.reindex(
        [x[:-2] for x in corrected_adata.obs.index]
    )

    # Reorder to match the original indexing
    corrected_adata = corrected_adata[mdata['mCG'].obs.index, :]

    results_dir = os.path.join(
        constants.DEFAULT_RESULTS_PATH,
        dataset + "_integration",
        args.query_sample,
        args.var_dim,
        args.method,
        str(args.random_seed),
    )

    os.makedirs(results_dir, exist_ok=True)

    np.save(
        os.path.join(results_dir, "latent_representations.npy"),
        corrected_adata.X
    )

elif args.method == 'combat':
    sc.pp.scale(mdata['mCG'])
    sc.pp.scale(mdata['mCH'])

    sc.pp.pca(mdata['mCG'])
    sc.pp.pca(mdata['mCH'])

    adata_full = sc.AnnData(
        X=np.concatenate([
            normalize(mdata['mCG'].obsm['X_pca']),
            normalize(mdata['mCH'].obsm['X_pca'])
        ], axis=1),
        obs=mdata['mCG'].obs
    )

    results_dir = os.path.join(
        constants.DEFAULT_RESULTS_PATH,
        dataset + "_integration",
        args.query_sample,
        args.var_dim,
        args.method,
        str(args.random_seed),
    )

    os.makedirs(results_dir, exist_ok=True)

    sc.pp.combat(adata_full, key=batch_key)

    np.save(
        os.path.join(results_dir, "latent_representations.npy"),
        adata_full.X
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
        adata_full[adata_full.obs[batch_key] == x] for x in adata_full.obs[batch_key].unique()
    ]

    min_sample = adata_full.obs[batch_key].value_counts().min()
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
        dataset + "_integration",
        args.query_sample,
        args.var_dim,
        args.method,
        str(args.random_seed),
    )

    os.makedirs(results_dir, exist_ok=True)

    np.save(
        os.path.join(results_dir, "latent_representations.npy"),
        corrected_adata.obsm['X_pca_integrate']
    )