import argparse
import os
import sys

import pandas as pd
from ALLCools.mcds import MCDS
from utils import download_binary_file, mcds_to_anndata
from mudata import MuData

parser = argparse.ArgumentParser()

parser.add_argument(
    "--region",
    choices=[
        "HPF",
        "Isocortex",
        "OLF",
        "CNU"
    ],
)

parser.add_argument(
    "--subregion",
)

args = parser.parse_args()
print(f"Running {sys.argv[0]} with arguments")
for arg in vars(args):
    print(f"\t{arg}={getattr(args, arg)}")

region = args.region
subregion = args.subregion


hdf5_path = "/projects/leelab2/metVI/Liu2021/data"
os.makedirs(hdf5_path, exist_ok=True)

if subregion is not None:
    data_path = f"/projects/leelab2/metVI/Liu2021_{region}_{subregion}/data"
else:
    data_path = f"/projects/leelab2/metVI/Liu2021_{region}/data"

os.makedirs(data_path, exist_ok=True)

cell_metadata_file_url = "https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-020-03182-8/MediaObjects/41586_2020_3182_MOESM9_ESM.xlsx"

download_binary_file(
    file_url=cell_metadata_file_url,
    output_path=os.path.join(data_path, "cell_metadata.xlsx"),
)

metadata = pd.read_excel(
    os.path.join(data_path, "cell_metadata.xlsx"), index_col=0, header=15
)

region_metadata = metadata[metadata["MajorRegion"] == args.region]
if args.subregion is not None:
    region_metadata = region_metadata[region_metadata["SubRegion"] == subregion]
sample_list = region_metadata["Sample"].unique()
sample_list = [x.replace("_", "-") for x in sample_list]

file_url_list = [
    f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE132nnn/GSE132489/suppl/GSE132489_{sample}.mcds.hdf5" for sample in sample_list
]

for file_url in file_url_list:
    download_binary_file(
        file_url=file_url,
        output_path=os.path.join(hdf5_path, file_url.split("GSE132489_")[1]),
    )

mcds_paths = [os.path.join(hdf5_path, x.split("GSE132489_")[1]) for x in file_url_list]
for path in mcds_paths:
    os.system(f"allcools convert-mcds-to-zarr {path}")

# Dimension name used to do clustering
# This corresponding to AnnData .obs and .var
obs_dim = "cell"  # observation

var_dim = "gene"
mcds = MCDS.open(
    mcds_paths, obs_dim=obs_dim, var_dim=var_dim, use_obs=region_metadata.index
)

mcds.add_feature_cov_mean(var_dim=var_dim)

blacklist_file_url = (
    "https://github.com/Boyle-Lab/Blacklist/raw/master/lists/mm10-blacklist.v2.bed.gz"
)
download_binary_file(
    file_url=blacklist_file_url,
    output_path=os.path.join(data_path, blacklist_file_url.split("/")[-1]),
)

exclude_chromosome = ["chrM", "chrY"]

# filter by coverage - based on the distribution above
min_cov = 100
if var_dim == "chrom100k":
    mcds = mcds.filter_feature_by_cov_mean(
        min_cov=min_cov,
    )
elif var_dim == "gene":
    mcds = mcds.filter_feature_by_cov_mean(
        min_cov=min_cov,
    )

# Features having overlap > f with any black list region will be removed.
black_list_fraction = 0.2
mcds = mcds.remove_black_list_region(
    black_list_path=os.path.join(data_path, blacklist_file_url.split("/")[-1]),
    f=black_list_fraction,
)

# remove chromosomes
mcds = mcds.remove_chromosome(exclude_chromosome)

mcds.add_mc_frac(
    normalize_per_cell=True,  # after calculating mC frac, per cell normalize the matrix
    clip_norm_value=10,  # clip outlier values above 10 to 10
)

# load only the mC fraction matrix into memory so following steps is faster
# Only load into memory when your memory size is enough to handle your dataset
if mcds.get_index(obs_dim).size < 20000:
    mcds[f"{var_dim}_da_frac"].load()

# HVF
mch_pattern = "CHN"
mcg_pattern = "CGN"
n_top_feature = 2500

# PC cutoff
pc_cutoff = 0.1

mch_hvf = mcds.calculate_hvf_svr(
    var_dim=var_dim, mc_type=mch_pattern, n_top_feature=n_top_feature, plot=False
)

mcg_hvf = mcds.calculate_hvf_svr(
    var_dim=var_dim, mc_type=mcg_pattern, n_top_feature=n_top_feature, plot=False
)

mcg_adata = mcds_to_anndata(
    mcds, "CGN", var_dim=var_dim, obs_dim=obs_dim, metadata=metadata, subset_features=True
)

mch_adata = mcds_to_anndata(
    mcds, "CHN", var_dim=var_dim, obs_dim=obs_dim, metadata=metadata, subset_features=True
)

mdata = MuData({"mCG": mcg_adata, "mCH": mch_adata})
mdata.write_h5mu(
    os.path.join(data_path, f"{var_dim}_{n_top_feature}_features.h5mu")
)
