import os

import mudata
import scanpy as sc
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from scib import metrics
from sklearn.preprocessing import normalize
import pickle
import argparse
import sys
from constants import batch_keys, label_keys

parser = argparse.ArgumentParser()

parser.add_argument(
    "query_sample",
    type=str,
    choices=[
        "UMB4540_hs_25yr",
        "NDARKD326LNK_hs_25yr",
        "NDARKJ183CYT_hs_58yr",
        "mCTseq_hs_21yr",
        "mCTseq_hs_29yr",
        "snm3Cseq_hs_21yr",
        "snm3Cseq_hs_29yr"
    ],
    help="Which dataset to use as a query sample"
)

parser.add_argument(
    "method",
    type=str,
    choices=[
        "methylANVI_arches",
        "methylANVI_denovo",
        "methylVI_arches",
        "methylVI_denovo",
        "harmony",
        "scanorama",
        "mnn",
        "seurat",
        "combat",
        "unintegrated"
    ],
    help="Which method to use"
)
args = parser.parse_args()

mdata = mudata.read_h5mu(
    f"/projects/leelab3/methylVI/Luo2022_atlas/data/gene_2500_features.h5mu"
)
#batch_correction_metrics_dict = defaultdict(defaultdict(int).copy)
#bio_conservation_metrics_dict = defaultdict(defaultdict(int).copy)

mdata['mCG'].obsm['methylVI_arches'] = np.load(
    f"/projects/leelab2/metVI/results/Luo2022_atlas_integration/{args.query_sample}/gene/methylVI_arches/likelihood_betabinomial/dispersion_gene/latent_20/123/latent_representations.npy"
)

mdata['mCG'].obsm['methylANVI_arches'] = np.load(
    f"/projects/leelab2/metVI/results/Luo2022_atlas_integration/{args.query_sample}/gene/methylANVI_arches/likelihood_betabinomial/dispersion_gene/latent_20/123/latent_representations.npy"
)

mdata['mCG'].obsm['methylVI_denovo'] = np.load(
    f"/projects/leelab2/metVI/results/Luo2022_atlas_integration/{args.query_sample}/gene/methylVI_denovo/likelihood_betabinomial/dispersion_gene/latent_20/123/latent_representations.npy"
)

mdata['mCG'].obsm['methylANVI_denovo'] = np.load(
    f"/projects/leelab2/metVI/results/Luo2022_atlas_integration/{args.query_sample}/gene/methylANVI_denovo/likelihood_betabinomial/dispersion_gene/latent_20/123/latent_representations.npy"
)

mdata['mCG'].obsm['harmony'] = np.load(
    f"/projects/leelab2/metVI/results/Luo2022_atlas_integration/{args.query_sample}/gene/harmony/123/latent_representations.npy"
)

mdata['mCG'].obsm['scanorama'] = np.load(
    f"/projects/leelab2/metVI/results/Luo2022_atlas_integration/{args.query_sample}/gene/scanorama/123/latent_representations.npy"
)

mdata['mCG'].obsm['mnn'] = np.load(
    f"/projects/leelab2/metVI/results/Luo2022_atlas_integration/{args.query_sample}/gene/mnn/123/latent_representations.npy"
)

mdata['mCG'].obsm['seurat'] = np.load(
    f"/projects/leelab2/metVI/results/Luo2022_atlas_integration/{args.query_sample}/gene/seurat/123/latent_representations.npy"
)

sc.pp.scale(mdata['mCG'])
sc.pp.scale(mdata['mCH'])

sc.pp.pca(mdata['mCG'])
sc.pp.pca(mdata['mCH'])

mdata['mCG'].obsm['unintegrated'] = np.concatenate([
    mdata['mCG'].obsm['X_pca'] / mdata['mCG'].obsm['X_pca'].std(),
    mdata['mCH'].obsm['X_pca'] / mdata['mCG'].obsm['X_pca'].std()
], axis=1)

model = args.method
batch_correction_metrics_dict = {}
bio_conservation_metrics_dict = {}
sc.pp.neighbors(mdata['mCG'], use_rep=model)

batch_correction_metrics_dict["kBET"] = metrics.kBET(
    mdata['mCG'],
    batch_key=batch_keys["Luo2022_atlas"],
    label_key=label_keys["Luo2022_atlas"],
    type_="embed",
    embed=model
)

batch_correction_metrics_dict["Graph iLISI"] = metrics.ilisi_graph(
    mdata['mCG'],
    batch_key=batch_keys["Luo2022_atlas"],
    type_="embed",
    use_rep=model
)

batch_correction_metrics_dict['Batch ASW'] = metrics.silhouette_batch(
    mdata['mCG'],
    batch_key=batch_keys["Luo2022_atlas"],
    label_key=label_keys["Luo2022_atlas"],
    embed=model
)

batch_correction_metrics_dict['Graph connectivity'] = metrics.graph_connectivity(
    mdata['mCG'],
    label_key=label_keys["Luo2022_atlas"],
)

batch_correction_metrics_dict['PCR batch'] = metrics.pcr_comparison(
    mdata['mCG'],
    mdata['mCG'],
    covariate=batch_keys["Luo2022_atlas"],
    embed=model
)

metrics.cluster_optimal_resolution(
    mdata['mCG'],
    cluster_key="cluster",
    label_key=label_keys["Luo2022_atlas"],
    use_rep=model
)

bio_conservation_metrics_dict['Cell type ARI'] = metrics.ari(
    mdata['mCG'],
    cluster_key="cluster",
    label_key=label_keys["Luo2022_atlas"]
)

bio_conservation_metrics_dict['Cell type NMI'] = metrics.nmi(
    mdata['mCG'],
    cluster_key="cluster",
    label_key=label_keys["Luo2022_atlas"]
)

bio_conservation_metrics_dict['Cell type ASW'] = metrics.silhouette(
    mdata['mCG'],
    label_key=label_keys["Luo2022_atlas"],
    embed=model,
)

bio_conservation_metrics_dict['Isolated label F1'] = metrics.isolated_labels_f1(
    mdata['mCG'],
    label_keys["Luo2022_atlas"],
    batch_keys["Luo2022_atlas"],
    model,
)

bio_conservation_metrics_dict['Isolated label ASW'] = metrics.isolated_labels_asw(
    mdata['mCG'],
    label_keys["Luo2022_atlas"],
    batch_keys["Luo2022_atlas"],
    model,
)

bio_conservation_metrics_dict['Graph cLISI'] = metrics.clisi_graph(
    mdata['mCG'],
    label_key=label_keys["Luo2022_atlas"],
    type_='embed',
    use_rep=model,
)

os.makedirs(
    f"/projects/leelab3/methylVI/Luo2022_atlas_integration/{args.query_sample}/",
    exist_ok=True
)

pickle.dump(
    batch_correction_metrics_dict,
    open(f"/projects/leelab3/methylVI/Luo2022_atlas_integration/{args.query_sample}/{model}_batch_correction_metrics.pkl", "wb")
)

pickle.dump(
    bio_conservation_metrics_dict,
    open(f"/projects/leelab3/methylVI/Luo2022_atlas_integration/{args.query_sample}/{model}_bio_conservation_metrics.pkl", "wb")
)
