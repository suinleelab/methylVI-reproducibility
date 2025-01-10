import os
import shutil
import tarfile

import GEOparse
import pandas as pd
from ALLCools.count_matrix.dataset import generate_dataset
import numpy as np
from ALLCools.utilities import standardize_allc
from ALLCools.mcds import MCDS
from tqdm import tqdm
from utils import download_binary_file, mcds_to_anndata
from mudata import MuData

destdir = "/projects/leelab3/methylVI/Liu2021_snm3C-seq/data"
os.makedirs(destdir, exist_ok=True)

gse = GEOparse.get_GEO(geo="GSE156683", destdir=destdir, silent=True)

download_binary_file(
    "https://raw.githubusercontent.com/igvteam/igv/master/genomes/sizes/mm10.chrom.sizes",
    "/projects/leelab3/methylVI/Liu2021_snm3C-seq/mm10.chrom.sizes",
)

for _, gsm in tqdm(sorted(gse.gsms.items())):
    # If we're resuming from a previous run, then we skip any stuff that's already present
    if os.path.exists(
        os.path.join(destdir, gsm.metadata["title"][0].rsplit("P56-")[1], "allc_full.tsv.gz")
    ) and os.path.exists(
        os.path.join(destdir, gsm.metadata["title"][0].rsplit("P56-")[1], "allc_full.tsv.gz.tbi")
    ):
        continue
    tmp = gsm.download_supplementary_files(destdir, download_sra=False)

    # Grab the file location
    # Should be in the format /projects/leelab2/metVI/Luo2017_Human/data/Supp_GSMXXX_SampleName/Supp_GSMXXX_SampleName.tar.gz
    downloaded_allc_file_path = [x for x in tmp.values() if x.endswith(".tsv.gz")][0]

    # For each sample, we rename the allc file to "allc_full" to make future steps easier
    folder_path = downloaded_allc_file_path.rsplit("/", maxsplit=1)[0]
    os.rename(downloaded_allc_file_path, os.path.join(folder_path, "allc_full.tsv.gz"))

    # The folder has some unnecessary cruft in the directory name
    # Here we remove that cruft to match the sample ids in the metadata file
    new_folder_path = os.path.join(destdir, folder_path.rsplit("P56-", maxsplit=1)[1])
    os.rename(folder_path, new_folder_path)

    standardize_allc(
        allc_path=os.path.join(new_folder_path, "allc_full.tsv.gz"),
        chrom_size_path="/projects/leelab3/methylVI/Liu2021_snm3C-seq/mm10.chrom.sizes",
    )



download_binary_file(
    "https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-020-03182-8/MediaObjects/41586_2020_3182_MOESM12_ESM.xlsx",
    "/projects/leelab3/methylVI/Liu2021_snm3C-seq/metadata.xlsx",
)

# Create an ALLC table for downloaded cells. We'll use this to generate our MCDS file
# later from the command line.
metadata = pd.read_excel(
    "/projects/leelab3/methylVI/Liu2021_snm3C-seq/metadata.xlsx", index_col=0, header=1
)

# Sucessfully processed cells have an indexed (i.e., 'tbi' extension) ALLC file
processed_cells = [
    x for x in os.listdir(destdir) if os.path.isdir(os.path.join(destdir, x))
]
processed_cells = [
    x
    for x in processed_cells
    if os.path.exists(os.path.join(destdir, x, "allc_full.tsv.gz.tbi"))
]

# Create ALLC table with two columns: first has cell name, second has corresponding ALLC file location
allc_path_list = [os.path.join(destdir, x, "allc_full.tsv.gz") for x in processed_cells]
allc_table = pd.DataFrame(index=processed_cells, data={"files": allc_path_list})
overlapping_cells = [x for x in metadata.index if x in allc_table.index]
allc_table = allc_table.loc[overlapping_cells]
allc_table_path = "/projects/leelab3/methylVI/Liu2021_snm3C-seq/allc_table.tsv"
allc_table.to_csv(allc_table_path, header=False, sep="\t")

# Download and preprocess gene body annotation file for use by ALLCools.
# AWK command taken from endrebak's answer at https://www.biostars.org/p/56280/
gene_body_annotation_url = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M22/gencode.vM22.annotation.gtf.gz"
download_binary_file(
    gene_body_annotation_url,
    "/projects/leelab3/methylVI/Liu2021_snm3C-seq/gencode.vM22.annotation.gtf.gz",
)
os.chdir("/projects/leelab3/methylVI/Liu2021_snm3C-seq/")
os.system(
    'zcat gencode.vM22.annotation.gtf.gz | awk \'OFS="\t" {if ($3=="gene") {print $1,$4-1,$5-1,$10,$16,$7}}\' | tr -d \'";\' > mm10_gene_bodies.bed'
)

# Deletes information for badly formed chromosomes like chr1_XXX_random
os.system(
    "sed -i '/^chr.*_/d' mm10.chrom.sizes"
)

# Create MCDS file with all desired regions
mcds_output_path = "/projects/leelab3/methylVI/Liu2021_snm3C-seq/Liu2021_snm3C-seq.mcds"
allc_table_path = "/projects/leelab3/methylVI/Liu2021_snm3C-seq/allc_table.tsv"
os.chdir("/projects/leelab3/methylVI/Liu2021_snm3C-seq/")
if os.path.exists(mcds_output_path):
    print("MCDS file already present. Skipping MCDS file generation...")
else:
    os.system(
        f"allcools generate-dataset \
        --allc_table allc_table.tsv \
        --output_path Liu2021_snm3C-seq.mcds \
        --chrom_size_path mm10.chrom.sizes \
        --obs_dim cell \
        --cpu 50 \
        --chunk_size 50 \
        --regions chrom100k 100000 \
        --regions gene mm10_gene_bodies.bed \
        --quantifiers chrom100k count CGN,CHN \
        --quantifiers gene count CGN,CHN"
    )

mcds = MCDS.open(
    "Liu2021_snm3C-seq.mcds",
    obs_dim="cell",
    var_dim="chrom100k",
)
mcds  = mcds.rename({
    "chrom100k_end": "chrom100k_bin_end",
    "chrom100k_start": "chrom100k_bin_start",
})
mcds = mcds.assign_coords({
    "chrom100k": np.arange(len(mcds.get(f"chrom100k_chrom")))
})
mcds.write_dataset(
    output_path="Liu2021_snm3C-seq_chrom100k.mcds",
    mode="w-",
    obs_dim="cell",
    var_dims="chrom100k"
)

os.system(
    "rm -rf Liu2021_snm3C-seq.mcds/chrom100k"
)

os.system(
    "mv Liu2021_snm3C-seq_chrom100k.mcds/chrom100k Liu2021_snm3C-seq.mcds/chrom100k"
)

# Read in generated MCDS file
for var_dim in ["chrom100k", "gene"]:
    num_features = 2500
    obs_dim = "cell"
    mcds = MCDS.open(mcds_output_path, obs_dim=obs_dim, var_dim=var_dim)

    # Feature cov cutoffs. Taken from ALLCools tutorial at
    # https://lhqing.github.io/ALLCools/cell_level/basic/mch_mcg_100k_basic.html
    min_cov = 500
    max_cov = 3000

    mcds.add_feature_cov_mean(var_dim=var_dim)

    if var_dim == "chrom100k":
        # Filter by coverage - based on the distribution above
        mcds = mcds.filter_feature_by_cov_mean(
            min_cov=min_cov, max_cov=max_cov  # minimum coverage  # maximum coverage
        )

    if var_dim == "gene":
        min_cov = 100
        mcds = mcds.filter_feature_by_cov_mean(
            min_cov=min_cov,
        )

    # Remove blacklist regions for mice. These regions are known to have data quality issues
    # and will be removed from further analysis.
    blacklist_file_url = "https://github.com/Boyle-Lab/Blacklist/raw/master/lists/mm10-blacklist.v2.bed.gz"
    download_binary_file(
        file_url=blacklist_file_url,
        output_path=os.path.join(destdir, blacklist_file_url.split("/")[-1]),
    )

    # Features having overlap > f with any black list region will be removed.
    black_list_fraction = 0.2
    mcds = mcds.remove_black_list_region(
        black_list_path=os.path.join(destdir, blacklist_file_url.split("/")[-1]),
        f=black_list_fraction,
    )

    # Remove problematic chromosomes
    exclude_chromosome = ["chrM", "chrY"]
    mcds = mcds.remove_chromosome(exclude_chromosome)

    mcds.add_mc_frac(
        normalize_per_cell=True,  # after frac, per cell normalize the matrix
        clip_norm_value=10,  # clip outlier values above 10 to 10
    )

    # load only the mC fraction matrix into memory so following steps is faster
    # Only load into memory when your memory size is enough to handle your dataset
    if mcds.get_index(obs_dim).size < 20000:
        mcds[f"{var_dim}_da_frac"].load()

    # Select highly variable features
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
    mdata.write_h5mu(
        os.path.join(destdir, f"{var_dim}_{num_features}_features.h5mu")
    )