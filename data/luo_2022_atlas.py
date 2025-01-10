import pandas as pd
from mudata import MuData
from data.utils import download_binary_file
import os
import xarray as xr
from ALLCools.mcds import MCDS
from data.utils import mcds_to_anndata


destdir = "/projects/leelab3/methylVI/Luo2022_atlas/data"
os.makedirs(destdir, exist_ok=True)

download_binary_file(
    "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9004682/bin/mmc7.xlsx",
    os.path.join(destdir, "cell_metadata.xlsx"),
)

# Here we clean up metadata cell IDs from Excel file to match what's present in the MCDS file
metadata = pd.read_excel(os.path.join(destdir, "cell_metadata.xlsx"), header=2)
metadata['sample'] = [x.split("_")[1] for x in metadata['Cell ID']]


# For cells sampled with snm3C-seq, the metadata index has "_indexed" at the end, while the MCDS does not
# Here we remove the extra "_indexed" for consistency
metadata['Cell ID'] = [x.split("_indexed")[0] if y == "snm3Cseq" else x for x, y in
                       zip(metadata['Cell ID'], metadata["sample"])]

# The sample ids here ("NDARKD326LNK" and "NDARKJ183CYT") are reversed in the metadata compared to the MCDS files
# Here we swap their occurances in the metadata file to achieve consistency
metadata['Cell ID'] = [
    x.replace("NDARKD326LNK", "NDARKJ183CYT") if y == "NDARKD326LNK" else x for x, y in
    zip(metadata['Cell ID'], metadata["sample"])
]
metadata['Cell ID'] = [
    x.replace("NDARKJ183CYT", "NDARKD326LNK", ) if y == "NDARKJ183CYT" else x for x, y in
    zip(metadata['Cell ID'], metadata["sample"])
]

# The sample ids with "NDARKD326LNK" have an extra "NA" term in the MCDS file after the FCXX term that's not present in the metadata file
# Here we add that NA term in for consistency
cell_ids_final = []
for cell_id in metadata['Cell ID']:
    if "NDARKD326LNK" in cell_id:
        cell_id_chunks = cell_id.rsplit("_", maxsplit=3)
        cell_id_chunks.insert(1, "NA")
        cell_ids_final.append("_".join(cell_id_chunks))
    else:
        cell_ids_final.append(cell_id)

metadata['Cell ID'] = cell_ids_final

# Refresh the sample ids to account for fixes in lines above
metadata['sample'] = [f"{x.split('_')[1]}_{x.split('_')[2]}_{x.split('_')[3]}" for x in metadata['Cell ID']]
metadata.index = metadata['Cell ID']

obs_dim = "cell"

for var_dim in ["gene", "chrom100k"]:
    num_features = 2500
    mcds_snmC2T = MCDS.open(
        "/projects/leelab3/methylVI/Luo2022_snmCAT-seq/data/snmC2T-seq.GEO.mcds",
        obs_dim=obs_dim,
        var_dim=var_dim,
        use_obs=metadata.index
    )

    mcds = MCDS.open(
        [
            "/projects/leelab3/methylVI/Luo2022_snmCAT-seq/data/snm3C-seq.GEO.mcds",
            "/projects/leelab3/methylVI/Luo2022_snmCAT-seq/data/snmC-seq_and_snmC-seq2.GEO.mcds",
        ],
        obs_dim=obs_dim,
        var_dim=var_dim,
        use_obs=metadata.index
    )

    # CGN/CHN methylation in the snmC2T dataset are labelled as HCGN/HCHN. Here we modify these
    # labels to CGN/CHN for consistency with other datasets.
    mcds_snmC2T['mc_type'] = mcds_snmC2T.mc_type.where(mcds_snmC2T.mc_type != 'HCGN', 'CGN')
    mcds_snmC2T['mc_type'] = mcds_snmC2T.mc_type.where(mcds_snmC2T.mc_type != 'HCHN', 'CHN')

    # Drop GCYN data since we won't use it.
    mcds_snmC2T = mcds_snmC2T.where(mcds_snmC2T.mc_type != "GCYN", drop=True)

    # Only the snmC2T MCDS file has proper coordinate information for gene/chrom100k coords.
    # Here we transfer this information to the other datasets.
    mcds.coords['gene'] = mcds_snmC2T.coords['gene']
    mcds.coords['chrom100k'] = mcds_snmC2T.coords['chrom100k']

    mcds_snmC2T = mcds_snmC2T.drop_vars("rna_da")
    mcds = xr.concat([mcds_snmC2T, mcds], dim="cell")
    mcds = MCDS(mcds, obs_dim=obs_dim, var_dim=var_dim)

    # We have some CHN measurements with absurdly high coverage, which lead to numerical stability issues
    # later. Here we assume those reads are low quality and remove them.
    min_cov = 100
    max_cov = 15000
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
    mcds = mcds.filter_feature_by_cov_mean(
        min_cov=min_cov, max_cov=max_cov  # minimum coverage  # maximum coverage
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

    mch_pattern = "CHN"
    mcg_pattern = "CGN"

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

    mdata = MuData({"mCG": mcg_adata, "mCH": mch_adata})
    mdata = mdata[mdata.obs['mCG:MajorType'] != "Outlier"]

    mdata.write_h5mu(
        os.path.join(destdir, f"{var_dim}_{num_features}_features.h5mu")
    )