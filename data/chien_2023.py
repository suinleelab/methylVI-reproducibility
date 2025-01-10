import os
import shutil
import tarfile

import GEOparse
import pandas as pd
from ALLCools.count_matrix.dataset import generate_dataset
from ALLCools.mcds import MCDS
from ALLCools.utilities import standardize_allc
import pandas as pd
import scanpy as sc
from tqdm import tqdm
from utils import download_binary_file, make_obs_df_var_df, mcds_to_anndata
from mudata import MuData

destdir = "/data/Chien2023/data"
os.makedirs(destdir, exist_ok=True)

# Get mouse human size file for preprocessing
chrom_size_path = "/data/Chien2023/hg38.chrom.sizes"
download_binary_file(
    "https://raw.githubusercontent.com/igvteam/igv/master/genomes/sizes/hg38.chrom.sizes",
    chrom_size_path,
)

gse_list = [
    "GSE193274",
    "GSE193296",
    "GSE193299",
    "GSE193313",
    "GSE193339",
    "GSE193372",
    "GSE193458",
    "GSE193499",
    "GSE201248",
    "GSE201830",
    "GSE201910",
    "GSE201933",
    "GSE202033",
    "GSE202062",
    "GSE202125",
    "GSE202162",
    "GSE249238",
    "GSE249336",
    "GSE249399",
    "GSE249705"
]

for gse_number in gse_list:
    gse = GEOparse.get_GEO(
        geo=gse_number,
        destdir=destdir,
        silent=True
    )

    for _, gsm in tqdm(sorted(gse.gsms.items())):
        species = gsm.metadata["organism_ch1"][0]
        title = gsm.metadata["title"][0]

        if os.path.exists(
            os.path.join(destdir, gse_number, title,  "allc_full.tsv.gz")
        ) and os.path.exists(
            os.path.join(destdir, gse_number, title, "allc_full.tsv.gz.tbi")
        ):
            continue

        tmp = gsm.download_supplementary_files(destdir, download_sra=False)

        old_allc_file_name = list(tmp.values())[0]
        old_allc_file_dir = old_allc_file_name.rsplit("/", maxsplit=1)[0]

        new_allc_file_dir = os.path.join(destdir, gse_number, title)
        os.makedirs(new_allc_file_dir)
        new_allc_file_name = os.path.join(new_allc_file_dir, "allc_full.tsv.gz")

        os.rename(
            old_allc_file_name,
            new_allc_file_name
        )

        os.rmdir(old_allc_file_dir)

        standardize_allc(
            allc_path=new_allc_file_name,
            chrom_size_path=chrom_size_path,
            remove_additional_chrom=True
        )

# Create ALLC table files for each GSE entry
for gse_number in gse_list:
    allc_dir = os.path.join("/data", "Chien2023", "data", gse_number)

    # Create an ALLC table for downloaded cells. We'll use this to generate our MCDS file
    # later from the command line.

    # Sucessfully processed cells have an indexed (i.e., 'tbi' extension) ALLC file
    processed_cells = [
        x for x in os.listdir(allc_dir) if os.path.isdir(os.path.join(allc_dir, x))
    ]
    processed_cells = [
        x
        for x in processed_cells
        if os.path.exists(os.path.join(allc_dir, x, "allc_full.tsv.gz.tbi"))
    ]

    # Create ALLC table with two columns: first has cell name, second has corresponding ALLC file location
    allc_path_list = [os.path.join(allc_dir, x, "allc_full.tsv.gz") for x in processed_cells]
    allc_table = pd.DataFrame(index=processed_cells, data={"files": allc_path_list})
    allc_table_path = f"/data/Chien2023/{gse_number}_allc_table.tsv"

    allc_table.to_csv(allc_table_path, header=False, sep="\t")

# Download and preprocess gene body annotation file for use by ALLCools.
# AWK command taken from endrebak's answer at https://www.biostars.org/p/56280/
gene_body_annotation_url = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_38/gencode.v38.annotation.gtf.gz"
download_binary_file(
    gene_body_annotation_url,
    "/data/Chien2023/gencode.v38.annotation.gtf.gz",
)
os.chdir("/data/Chien2023/")
os.system(
    'zcat gencode.v38.annotation.gtf.gz | awk \'OFS="\t" {if ($3=="gene") {print $1,$4-1,$5,$10,$16,$7}}\' | tr -d \'";\' > hg38_gene_bodies.bed'
)

os.system(
    'zcat gencode.v38.annotation.gtf.gz | awk \'OFS="\t" {if ($3=="gene") {print $1,$4-1,$5,$10,$16,$7}}\' | tr -d \'";\' > hg38_gene_bodies_promoters.bed'
)

bed = pd.read_csv("hg38_gene_bodies_promoters.bed", sep="\t", header=None)
bed[bed[1] < 0] = 0
bed.to_csv("hg38_gene_bodies_promoters.bed", index=False, header=False, sep="\t")

# Create MCDS files for each GSE entry
for gse_number in gse_list:
    mcds_output_path = f"/data/Chien2023/{gse_number}.mcds"
    allc_table_path = f"/data/Chien2023/{gse_number}_allc_table.tsv"
    os.chdir("/data/Chien2023/")

    # Create MCDS file with all desired regions
    if os.path.exists(mcds_output_path):
        print(f"{gse_number} MCDS file already present. Skipping MCDS file generation...")
    else:
        print(f"Generating {gse_number} MCDS file")
        os.system(
            f"allcools generate-dataset \
            --allc_table {gse_number}_allc_table.tsv \
            --output_path {gse_number}.mcds \
            --chrom_size_path hg38.chrom.sizes \
            --obs_dim cell \
            --cpu 50 \
            --chunk_size 50 \
            --regions chrom100k 100000 \
            --regions gene hg38_gene_bodies.bed \
            --regions promoter hg38_gene_bodies_promoters.bed \
            --quantifiers chrom100k count CGN,CHN \
            --quantifiers gene count CGN,CHN \
            --quantifiers promoter count CGN,CHN"
        )

mcds_paths = [
    os.path.join("/data", "Chien2023", f"{gse_number}.mcds") for gse_number in gse_list
]

metadata = pd.read_excel("/data/Chien2023/cell_metadata.xlsx", index_col=0)
metadata = metadata[metadata['DNA_passQC']]

mudata_path = "/projects/leelab3/methylVI/Chien2023/data"
os.makedirs(mudata_path, exist_ok=True)

from utils import get_hs_ensembl_mappings
biomart_dict = get_hs_ensembl_mappings()

# Read in generated MCDS files
obs_dim = "cell"
for var_dim in ["gene", "promoter"]:
    num_features = 2500
    mcds = MCDS.open(
        mcds_paths, obs_dim=obs_dim, var_dim=var_dim, use_obs=metadata.index
    )

    mcds.add_feature_cov_mean(var_dim=var_dim)

    # Filter by coverage - based on the distribution above
    if var_dim == "chrom100k":
        min_cov = 100
        mcds = mcds.filter_feature_by_cov_mean(
            min_cov=min_cov,
        )
    elif var_dim == "gene":
        min_cov = 100
        mcds = mcds.filter_feature_by_cov_mean(
            min_cov=min_cov
        )

    blacklist_file_url = (
        "https://github.com/Boyle-Lab/Blacklist/raw/master/lists/hg38-blacklist.v2.bed.gz"
    )
    download_binary_file(
        file_url=blacklist_file_url,
        output_path=os.path.join("/data", "Chien2023", blacklist_file_url.split("/")[-1]),
    )

    # Features having overlap > f with any black list region will be removed.
    black_list_fraction = 0.2
    mcds = mcds.remove_black_list_region(
        black_list_path=os.path.join("/data", "Chien2023", blacklist_file_url.split("/")[-1]),
        f=black_list_fraction,
    )

    # remove problematic chromosomes
    exclude_chromosome = ["chrM", "chrY", "chrX"]
    mcds = mcds.remove_chromosome(exclude_chromosome)

    mcds.add_mc_frac(
        normalize_per_cell=True,  # after calculating mC frac, per cell normalize the matrix
        clip_norm_value=10,  # clip outlier values above 10 to 10
    )

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
    mdata = mdata[mdata.obs['mCH:level1'].notnull()]

    mdata.write_h5mu(
        os.path.join(mudata_path, f"{var_dim}_{num_features}_features.h5mu")
    )