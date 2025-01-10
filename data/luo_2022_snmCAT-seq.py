import os
import tarfile

import pandas as pd
from ALLCools.mcds import MCDS
from data.utils import download_binary_file, mcds_to_anndata
from mudata import MuData
import scanpy as sc

destdir = "/projects/leelab3/methylVI/Luo2022_snmCAT-seq/data"
os.makedirs(destdir, exist_ok=True)

cell_metadata_file_url = "https://ars.els-cdn.com/content/image/1-s2.0-S2666979X22000271-mmc5.xlsx"
download_binary_file(
    file_url=cell_metadata_file_url,
    output_path=os.path.join(destdir, "cell_metadata.xlsx"),
)

mcds_url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE140nnn/GSE140493/suppl/GSE140493_MCDS_data.tar.gz"
mcds_tar_file_path = os.path.join(destdir, mcds_url.split("GSE140493_")[1])
download_binary_file(
    file_url=mcds_url,
    output_path=mcds_tar_file_path
)

tar_file_object = tarfile.open(mcds_tar_file_path)
tar_file_object.extractall(destdir)
tar_file_object.close()

mcds_file_path = os.path.join(destdir, "snmC2T-seq.GEO.mcds")

rna_counts_url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE140nnn/GSE140493/suppl/GSE140493_snmC2T-seq.rna_gene_counts.csv.gz"
rna_counts_file_path = os.path.join(destdir, rna_counts_url.split("GSE140493_")[1])
download_binary_file(
    file_url=rna_counts_url,
    output_path=rna_counts_file_path
)

# Dimension name used to do clustering
# This corresponding to AnnData .obs and .var
obs_dim = "cell"  # observation

for var_dim in ["chrom100k", "gene"]:
    num_features = 2500
    mcds = MCDS.open(
        mcds_file_path, obs_dim=obs_dim, var_dim=var_dim
    )
    rna_counts = pd.read_csv(rna_counts_file_path, index_col=0)

    metadata = pd.read_excel(os.path.join(destdir, "cell_metadata.xlsx"), header=1, index_col=0)
    cells_to_keep = [x for x in mcds['cell'].to_pandas().index if x in metadata.index.intersection(rna_counts.index)]
    metadata = metadata.loc[cells_to_keep]
    mcds = mcds.loc[{"cell": metadata.index}]

    rna_counts = rna_counts.loc[cells_to_keep]
    rna_adata = sc.AnnData(X=rna_counts, obs=metadata)

    if num_features != "all":
        sc.pp.highly_variable_genes(rna_adata, n_top_genes=num_features, flavor='seurat_v3', subset=True)
    rna_adata.layers['counts'] = rna_adata.X.copy()
    sc.pp.normalize_total(rna_adata)
    sc.pp.log1p(rna_adata)

    # feature cov cutoffs
    min_cov = 100

    mcds.add_feature_cov_mean(var_dim=var_dim)

    blacklist_file_url = (
        "https://github.com/Boyle-Lab/Blacklist/raw/master/lists/hg38-blacklist.v2.bed.gz"
    )
    download_binary_file(
        file_url=blacklist_file_url,
        output_path=os.path.join(destdir, blacklist_file_url.split("/")[-1]),
    )

    exclude_chromosome = ["chrM", "chrY"]

    # filter by coverage - based on the distribution above
    if var_dim == "chrom100k":
        mcds = mcds.filter_feature_by_cov_mean(
            min_cov=min_cov,
        )

    # Features having overlap > f with any black list region will be removed.
    black_list_fraction = 0.2
    mcds = mcds.remove_black_list_region(
        black_list_path=os.path.join(destdir, blacklist_file_url.split("/")[-1]),
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
    mch_pattern = "HCHN"
    mcg_pattern = "HCGN"

    if num_features != "all":
        mch_hvf = mcds.calculate_hvf_svr(
            var_dim=var_dim, mc_type=mch_pattern, n_top_feature=num_features, plot=False
        )

        mcg_hvf = mcds.calculate_hvf_svr(
            var_dim=var_dim, mc_type=mcg_pattern, n_top_feature=num_features, plot=False
        )

        mcg_adata = mcds_to_anndata(
            mcds, mcg_pattern, var_dim=var_dim, obs_dim=obs_dim, metadata=metadata, subset_features=True
        )

        mch_adata = mcds_to_anndata(
            mcds, mch_pattern, var_dim=var_dim, obs_dim=obs_dim, metadata=metadata, subset_features=True
        )
    else:
        mcg_adata = mcds_to_anndata(
            mcds, mcg_pattern, var_dim=var_dim, obs_dim=obs_dim, metadata=metadata, subset_features=False
        )

        mch_adata = mcds_to_anndata(
            mcds, mch_pattern, var_dim=var_dim, obs_dim=obs_dim, metadata=metadata, subset_features=False
        )

    mcg_adata.var_names = [x + "_mCG" for x in mcg_adata.var_names]
    mch_adata.var_names = [x + "_mCH" for x in mch_adata.var_names]

    mdata = MuData({"mCG": mcg_adata, "mCH": mch_adata, "RNA": rna_adata})

    mdata.write_h5mu(
        os.path.join(destdir, f"{var_dim}_{num_features}_features.h5mu")
    )
