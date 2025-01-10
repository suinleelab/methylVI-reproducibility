import os
import shutil
import tarfile

import GEOparse
import pandas as pd
from ALLCools.count_matrix.dataset import generate_dataset
from ALLCools.mcds import MCDS
from ALLCools.utilities import standardize_allc
import scanpy as sc
from tqdm import tqdm
from utils import download_binary_file, make_obs_df_var_df, mcds_to_anndata
from mudata import MuData

destdir = "/projects/leelab2/metVI/Luo2017_Mouse/data"
os.makedirs(destdir, exist_ok=True)

# Gets all the info about this dataset from the GEO
gse = GEOparse.get_GEO(geo="GSE97179", destdir=destdir, silent=True)

# Get mouse chromosome size file for preprocessing
chrom_size_path = "/projects/leelab2/metVI/Luo2017_Mouse/mm10.chrom.sizes"
download_binary_file(
    "https://raw.githubusercontent.com/igvteam/igv/master/genomes/sizes/mm10.chrom.sizes",
    chrom_size_path,
)

# One sex chromosome + 19 non-sex chromosomes
chromosome_list = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    "X",
    "Y",
]

for _, gsm in sorted(gse.gsms.items()):
    species = gsm.metadata["organism_ch1"][0]
    title = gsm.metadata["title"][0]
    if species == "Mus musculus":
        # If we're resuming from a previous run, then we skip any stuff that's already present
        if os.path.exists(
            os.path.join(destdir, gsm.metadata["title"][0], "allc_full.tsv")
        ) and os.path.exists(
            os.path.join(destdir, gsm.metadata["title"][0], "allc_full.tsv.tbi")
        ):
            continue

        # Skip the bulk SST methylome data
        if title.startswith("SST"):
            continue

        # Return a dictionary with a single entry {file url: downloaded file location}
        tmp = gsm.download_supplementary_files(destdir, download_sra=False)

        # Grab the file location
        # Should be in the format /projects/leelab2/metVI/Luo2017_Mouse/data/Supp_GSM<XXX>_SampleName/Supp_GSM<XXX>_SampleName.tar.gz
        downloaded_file_path = list(tmp.values())[0]

        # Extract tar contents
        (
            download_folder,
            tar_file_name,
        ) = downloaded_file_path.rsplit("/", maxsplit=1)
        tar_file_object = tarfile.open(downloaded_file_path)
        tar_file_object.extractall(download_folder)
        tar_file_object.close()

        # Get the extracted directory (there should be only one directory here)
        extracted_folder_path = [
            x
            for x in os.listdir(download_folder)
            if os.path.isdir(os.path.join(download_folder, x))
        ][0]
        print(extracted_folder_path)

        # Move extracted directory up one level for easier access later
        old_full_folder_path = os.path.join(download_folder, extracted_folder_path)
        if extracted_folder_path.startswith("nuclei"):
            extracted_folder_path = extracted_folder_path.replace("-", "_")
            new_full_folder_path = os.path.join(
                destdir,
                f"{extracted_folder_path.split('_')[0]}_{extracted_folder_path.split('_')[1]}",
            )
        else:
            new_full_folder_path = os.path.join(destdir, extracted_folder_path)

        os.rename(old_full_folder_path, new_full_folder_path)

        # Don't need the original folder created by GEOParse where the file was downloaded
        shutil.rmtree(download_folder)

        # ALLC file names have format <prefix>_chr.tar.gz
        allc_file_prefix = os.listdir(new_full_folder_path)[0].rsplit("_", maxsplit=1)[
            0
        ]

        # ALLC files were split by chromosome in the GEO. Here we stitch them back together.
        # We wrap this in a "try" block because occasionally a file is malformatted and we don't want to crash our whole program
        try:
            allc_per_chromosome_dfs = []
            for chromosome in chromosome_list:
                chromosome_allc_file_path = f"{os.path.join(new_full_folder_path, allc_file_prefix)}_{chromosome}.tsv.gz"
                allc_per_chromosome_dfs.append(
                    pd.read_csv(chromosome_allc_file_path, sep="\t")
                )

            allc_full = pd.concat(allc_per_chromosome_dfs)

            output_path = os.path.join(new_full_folder_path, "allc_full.tsv")
            allc_full.to_csv(output_path, header=False, sep="\t", index=False)
            standardize_allc(allc_path=output_path, chrom_size_path=chrom_size_path)
        except Exception as e:
            print(f"Error when processing {allc_file_prefix}")
            print(e)
            shutil.rmtree(new_full_folder_path)

# Create an ALLC table for downloaded cells. We'll use this to generate our MCDS file
# later from the command line.
metadata = pd.read_excel(
    "/projects/leelab2/metVI/Luo2017_Mouse/cell_metadata.xlsx", index_col=0, header=1
)

# Sucessfully processed cells have an indexed (i.e., 'tbi' extension) ALLC file
processed_cells = [
    x for x in os.listdir(destdir) if os.path.isdir(os.path.join(destdir, x))
]
processed_cells = [
    x
    for x in processed_cells
    if os.path.exists(os.path.join(destdir, x, "allc_full.tsv.tbi"))
]

# Create ALLC table with two columns: first has cell name, second has corresponding ALLC file location
allc_path_list = [os.path.join(destdir, x, "allc_full.tsv") for x in processed_cells]
allc_table = pd.DataFrame(index=processed_cells, data={"files": allc_path_list})
overlapping_cells = [x for x in metadata.index if x in allc_table.index]
allc_table = allc_table.loc[overlapping_cells]
allc_table_path = "/projects/leelab2/metVI/Luo2017_Mouse/allc_table.tsv"
allc_table.to_csv(allc_table_path, header=False, sep="\t")

# Download and preprocess gene body annotation file for use by ALLCools.
# AWK command taken from endrebak's answer at https://www.biostars.org/p/56280/
gene_body_annotation_url = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M22/gencode.vM22.annotation.gtf.gz"
download_binary_file(
    gene_body_annotation_url,
    "/projects/leelab2/metVI/Luo2017_Mouse/gencode.vM22.annotation.gtf.gz",
)
os.chdir("/projects/leelab2/metVI/Luo2017_Mouse/")
os.system(
    'zcat gencode.vM22.annotation.gtf.gz | awk \'OFS="\t" {if ($3=="gene") {print $1,$4-1,$5,$10,$16,$7}}\' | tr -d \'";\' > mm10_gene_bodies.bed'
)

# Create MCDS file with all desired regions
mcds_output_path = "/projects/leelab2/metVI/Luo2017_Mouse/Luo2017_Mouse.mcds"
if os.path.exists(mcds_output_path):
    print("MCDS file already present. Skipping MCDS file generation...")
else:
    print("Generating MCDS file")
    os.system(
        f"allcools generate-dataset \
        --allc_table allc_table.tsv \
        --output_path Luo2017_Mouse.mcds \
        --chrom_size_path mm10.chrom.sizes \
        --obs_dim cell \
        --cpu 50 \
        --chunk_size 50 \
        --regions chrom100k 100000 \
        --regions gene mm10_gene_bodies.bed \
        --quantifiers chrom100k count CGN,CHN \
        --quantifiers gene count CGN,CHN"
    )

# Read in generated MCDS file
for var_dim in ["chrom100k", "gene"]:
    num_features = 2500
    obs_dim = "cell"
    mcds = MCDS.open(mcds_output_path, obs_dim=obs_dim, var_dim=var_dim)

    mcds.add_feature_cov_mean(var_dim=var_dim)

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
        normalize_per_cell=True,
        clip_norm_value=10,
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

    mcg_adata.obs["FACS date"] = mcg_adata.obs["FACS date"].astype(str)
    mcg_adata.var_names = [x + "_mCG" for x in mcg_adata.var_names]

    mch_adata.obs["FACS date"] = mch_adata.obs["FACS date"].astype(str)
    mch_adata.var_names = [x + "_mCH" for x in mch_adata.var_names]

    mdata = MuData({"mCG": mcg_adata, "mCH": mch_adata})
    mdata.write_h5mu(
        os.path.join(destdir, f"{var_dim}_{num_features}_features.h5mu")
    )