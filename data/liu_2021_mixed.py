import os
import pandas as pd
from ALLCools.mcds import MCDS
from utils import download_binary_file, mcds_to_anndata
from mudata import MuData
import xarray as xr

metadata = pd.read_excel(
    os.path.join("/projects/leelab2/metVI/Liu2021_HPF/data", "cell_metadata.xlsx"), index_col=0, header=15
)

region_metadata = metadata[metadata["MajorRegion"] == "HPF"]
sample_list = region_metadata["Sample"].unique()
sample_list = [x.replace("_", "-") for x in sample_list]

file_url_list = [
    f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE132nnn/GSE132489/suppl/GSE132489_{sample}.mcds.hdf5" for sample in sample_list
]

hdf5_path = "/projects/leelab2/metVI/Liu2021/data"
mcds_paths = [os.path.join(hdf5_path, x.split("GSE132489_")[1]) for x in file_url_list]

snmcseq2_mcds_paths = mcds_paths
snm3cseq_mcds_path = "/projects/leelab3/methylVI/Liu2021_snm3C-seq/Liu2021_snm3C-seq.mcds"

snmcseq2_metadata = metadata[metadata['MajorRegion'] == "HPF"]
snm3Cseq_metadata = pd.read_excel(
    "/projects/leelab3/methylVI/Liu2021_snm3C-seq/metadata.xlsx", index_col=0, header=1
)
full_metadata = pd.concat([snmcseq2_metadata, snm3Cseq_metadata])

cell_list = [
    "DG",
    "CA1",
    "CA3",
    "MGC",
    "VLMC",
    "OLF",
    "EC+PC",
    "ODC",
    "ASC",
    "ANP",
    "OPC",
    "Inh"
]

full_metadata['CoarseType'] = [x if x in cell_list else "Others" for x in full_metadata['MajorType']]

full_metadata['CoarseType'][full_metadata['CellClass'] == "Inh"] = "Inh"
full_metadata['CoarseType'][full_metadata['MajorType'].str.startswith("CA3")] = "CA3"
full_metadata['CoarseType'][full_metadata['MajorType'].str.startswith("VLMC")] = "VLMC"
full_metadata['CoarseType'][full_metadata['MajorType'].str.startswith("DG")] = "DG"
full_metadata['CoarseType'][full_metadata['MajorType'].isin(["EC", "PC"])] = "EC+PC"

# Region --> snm-3C-seq
# RegionName --> snmcseq-2
# SubRegion --> snmcseq-2
full_metadata = full_metadata[full_metadata['CoarseType'] != "Others"]
full_metadata['Region'] = full_metadata['Region'].astype(str)
full_metadata['RegionName'] = full_metadata['RegionName'].astype(str)

full_metadata['Platform'] = ["snm-3C-seq" if x == 'nan' else "snmcseq-2" for x in full_metadata['RegionName']]
full_metadata['RegionName'][full_metadata['RegionName'] == 'nan'] = full_metadata['Region'][full_metadata['Region'] != 'nan']
full_metadata = full_metadata[full_metadata['RegionName'].str.startswith('DG')]

obs_dim = "cell"

for var_dim in ["gene"]:
    snmcseq2_mcds = MCDS.open(
        snmcseq2_mcds_paths,
        obs_dim=obs_dim,
        var_dim=var_dim,
        use_obs=full_metadata.index
    )

    if var_dim == "gene":
        snmcseq2_mcds = snmcseq2_mcds.drop_dims("chrom100k")

    snm3cseq_mcds = MCDS.open(
        snm3cseq_mcds_path,
        obs_dim=obs_dim,
        var_dim=var_dim,
        use_obs=full_metadata.index
    )

    mcds = xr.concat([snm3cseq_mcds, snmcseq2_mcds], dim=obs_dim)
    mcds = MCDS(mcds, obs_dim=obs_dim, var_dim=var_dim)

    data_path = "/projects/leelab3/methylVI/Liu2021_mixed/data"
    os.makedirs(data_path, exist_ok=True)

    blacklist_file_url = (
        "https://github.com/Boyle-Lab/Blacklist/raw/master/lists/mm10-blacklist.v2.bed.gz"
    )
    download_binary_file(
        file_url=blacklist_file_url,
        output_path=os.path.join(data_path, blacklist_file_url.split("/")[-1]),
    )

    exclude_chromosome = ["chrM", "chrY"]

    mcds.add_feature_cov_mean(var_dim=var_dim)

    # filter by coverage - based on the distribution above
    if var_dim == "chrom100k":
        min_cov = 100
        mcds = mcds.filter_feature_by_cov_mean(
            min_cov=min_cov,
        )
    elif var_dim == "gene":
        min_cov = 100
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
    exclude_chromosome = ["chrM", "chrY"]
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
        mcds, "CGN", var_dim=var_dim, obs_dim=obs_dim, metadata=full_metadata, subset_features=True
    )

    mch_adata = mcds_to_anndata(
        mcds, "CHN", var_dim=var_dim, obs_dim=obs_dim, metadata=full_metadata, subset_features=True
    )

    mcg_adata.obs['Pass QC'].fillna(True, inplace=True)
    mch_adata.obs['Pass QC'].fillna(True, inplace=True)

    mcg_adata.obs['Pass Contact QC'].fillna(True, inplace=True)
    mch_adata.obs['Pass Contact QC'].fillna(True, inplace=True)

    mcg_adata.var_names = [x + "_mCG" for x in mcg_adata.var_names]
    mch_adata.var_names = [x + "_mCH" for x in mch_adata.var_names]

    mcg_adata = mcg_adata[mcg_adata.obs['RegionName'].str.startswith('DG')]
    mch_adata = mch_adata[mch_adata.obs['RegionName'].str.startswith('DG')]

    mdata = MuData({'mCG': mcg_adata.copy(), 'mCH': mch_adata.copy()})
    mdata = mdata[~mdata['mCG'].obs['CoarseType'].isin(["Others"])].copy()

    mdata.obs = mdata.obs.dropna(axis=1).copy()
    mdata['mCG'].obs =  mdata['mCG'].obs.dropna(axis=1).copy()
    mdata['mCH'].obs = mdata['mCH'].obs.dropna(axis=1).copy()

    mdata['mCG'].obs['Pass Contact QC'] = mdata['mCG'].obs['Pass Contact QC'].astype(bool)
    mdata['mCH'].obs['Pass Contact QC'] = mdata['mCH'].obs['Pass Contact QC'].astype(bool)
    mdata.obs['Platform'] = mdata.obs['mCG:Platform']

    mdata.write_h5mu(os.path.join(data_path, f"{var_dim}_{n_top_feature}_features.h5mu"))
