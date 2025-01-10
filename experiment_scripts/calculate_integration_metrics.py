import mudata
import scanpy as sc
from collections import defaultdict
from tqdm import tqdm
import os
import numpy as np
from scib import metrics
import pickle
import  argparse
from sklearn.preprocessing import normalize
from constants import batch_keys, label_keys

parser = argparse.ArgumentParser()
parser.add_argument(
    "dataset",
    type=str,
    choices=["Liu2021_mixed", "Luo2022_mixed", "Tian2023_mixed"],
    help="Which dataset to use for the experiment"
)

parser.add_argument(
    "method",
    type=str,
    choices=[
        "methylVI",
        "methylANVI",
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
if args.dataset == "Tian2023_mixed":
    mdata = mudata.read_h5mu(
        f"/projects/leelab2/metVI/{args.dataset}/data/CaB_gene_2500_features.h5mu"
    )
else:
    mdata = mudata.read_h5mu(
        f"/projects/leelab3/methylVI/{args.dataset}/data/gene_2500_features.h5mu"
    )
#batch_correction_metrics_dict = defaultdict(defaultdict(int).copy)
#bio_conservation_metrics_dict = defaultdict(defaultdict(int).copy)

mdata['mCG'].obsm['methylVI'] = np.load(
    f"/projects/leelab2/metVI/results/{args.dataset}_integration/gene/methylVI/likelihood_betabinomial/dispersion_gene/latent_20/123/latent_representations.npy"
)

#mdata['mCG'].obsm['methylANVI'] = np.load(
#    f"/projects/leelab2/metVI/results/{args.dataset}_integration/gene/methylANVI/likelihood_betabinomial/dispersion_gene/latent_20/123/latent_representations.npy"
#)

mdata['mCG'].obsm['harmony'] = np.load(
    f"/projects/leelab2/metVI/results/{args.dataset}_integration/gene/harmony/123/latent_representations.npy"
)

mdata['mCG'].obsm['scanorama'] = np.load(
    f"/projects/leelab2/metVI/results/{args.dataset}_integration/gene/scanorama/123/latent_representations.npy"
)

mdata['mCG'].obsm['mnn'] = np.load(
    f"/projects/leelab2/metVI/results/{args.dataset}_integration/gene/mnn/123/latent_representations.npy"
)

mdata['mCG'].obsm['seurat'] = np.load(
    f"/projects/leelab2/metVI/results/{args.dataset}_integration/gene/seurat/123/latent_representations.npy"
)

sc.pp.pca(mdata['mCG'])
sc.pp.pca(mdata['mCH'])

mdata['mCG'].obsm['unintegrated'] = np.concatenate([
    mdata['mCG'].obsm['X_pca'] / mdata['mCG'].obsm['X_pca'].std(),
    mdata['mCH'].obsm['X_pca'] / mdata['mCH'].obsm['X_pca'].std()
], axis=1)

mdata['mCG'].obsm['combat'] = np.load(
    f"/projects/leelab2/metVI/results/{args.dataset}_integration/gene/combat/123/latent_representations.npy"
)

model = args.method
batch_correction_metrics_dict = {}
bio_conservation_metrics_dict = {}
sc.pp.neighbors(mdata['mCG'], use_rep=model)

batch_correction_metrics_dict["kBET"] = metrics.kBET(
    mdata['mCG'],
    batch_key=batch_keys[args.dataset],
    label_key=label_keys[args.dataset],
    type_="embed",
    embed=model
)

batch_correction_metrics_dict["Graph iLISI"] = metrics.ilisi_graph(
    mdata['mCG'],
    batch_key=batch_keys[args.dataset],
    type_="embed",
    use_rep=model
)

batch_correction_metrics_dict['Batch ASW'] = metrics.silhouette_batch(
    mdata['mCG'],
    batch_key=batch_keys[args.dataset],
    label_key=label_keys[args.dataset],
    embed=model
)

batch_correction_metrics_dict['Graph connectivity'] = metrics.graph_connectivity(
    mdata['mCG'],
    label_key=label_keys[args.dataset]
)

batch_correction_metrics_dict['PCR batch'] = metrics.pcr_comparison(
    mdata['mCG'],
    mdata['mCG'],
    covariate=batch_keys[args.dataset],
    embed=model
)

metrics.cluster_optimal_resolution(
    mdata['mCG'],
    cluster_key="cluster",
    label_key=label_keys[args.dataset],
    use_rep=model
)

bio_conservation_metrics_dict['Cell type ARI'] = metrics.ari(
    mdata['mCG'],
    cluster_key="cluster",
    label_key=label_keys[args.dataset]
)

bio_conservation_metrics_dict['Cell type NMI'] = metrics.nmi(
    mdata['mCG'],
    cluster_key="cluster",
    label_key=label_keys[args.dataset]
)

bio_conservation_metrics_dict['Cell type ASW'] = metrics.silhouette(
    mdata['mCG'],
    label_key=label_keys[args.dataset],
    embed=model,
)

bio_conservation_metrics_dict['Isolated label F1'] = metrics.isolated_labels_f1(
    mdata['mCG'],
    label_keys[args.dataset],
    batch_keys[args.dataset],
    model,
)

bio_conservation_metrics_dict['Isolated label ASW'] = metrics.isolated_labels_asw(
    mdata['mCG'],
    label_keys[args.dataset],
    batch_keys[args.dataset],
    model,
)

bio_conservation_metrics_dict['Graph cLISI'] = metrics.clisi_graph(
    mdata['mCG'],
    label_key=label_keys[args.dataset],
    type_='embed',
    use_rep=model,
)

os.makedirs(f"/projects/leelab3/methylVI/{args.dataset}", exist_ok=True)

pickle.dump(
    batch_correction_metrics_dict,
    open(f"/projects/leelab3/methylVI/{args.dataset}/{model}_batch_correction_metrics.pkl", "wb")
)

pickle.dump(
    bio_conservation_metrics_dict,
    open(f"/projects/leelab3/methylVI/{args.dataset}/{model}_bio_conservation_metrics.pkl", "wb")
)
