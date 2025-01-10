from tqdm import tqdm
from data.utils import download_binary_file, mcds_to_anndata
import os
import shutil
import pandas as pd
from ALLCools.utilities import standardize_allc
from ALLCools.mcds import MCDS
from mudata import MuData

destdir = "/projects/leelab2/metVI/Mo_2015_MethylC-seq/data"

os.makedirs(destdir, exist_ok=True)

url_list = [
    "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM1541958&format=file&file=GSM1541958_allC.MethylC-seq_excitatory_neurons_rep1.tar.gz", #GSM1541958_allC.MethylC-seq_excitatory_neurons_rep1.tar.gz
    "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM1541959&format=file&file=GSM1541959_allC.MethylC-seq_excitatory_neurons_rep2.tar.gz", #GSM1541959_allC.MethylC-seq_excitatory_neurons_rep2.tar.gz
    "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM1541960&format=file&file=GSM1541960_allC.MethylC-seq_PV_neurons_rep1.tar.gz", #GSM1541960_allC.MethylC-seq_PV_neurons_rep1.tar.gz
    "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM1541961&format=file&file=GSM1541961_allC.MethylC-seq_PV_neurons_rep2.tar.gz", #GSM1541961_allC.MethylC-seq_PV_neurons_rep2.tar.gz
    "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM1541962&format=file&file=GSM1541962_allC.MethylC-seq_VIP_neurons_rep1.tar.gz", #GSM1541962_allC.MethylC-seq_VIP_neurons_rep1.tar.gz
    "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM1541963&format=file&file=GSM1541963_allC.MethylC-seq_VIP_neurons_rep2.tar.gz", #GSM1541963_allC.MethylC-seq_VIP_neurons_rep2.tar.gz
]

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

chrom_size_path = "/projects/leelab2/metVI/Mo_2015_MethylC-seq/mm10.chrom.sizes"
download_binary_file(
    "https://raw.githubusercontent.com/igvteam/igv/master/genomes/sizes/mm10.chrom.sizes",
    chrom_size_path,
)

for url in url_list:
    tar_file_name = url.split("MethylC-seq_")[1]
    output_path = os.path.join(destdir, tar_file_name)
    extracted_output_path = output_path.split(".tar.gz")[0]

    # Skip any files that have already been downloaded + processed
    if os.path.exists(os.path.join(extracted_output_path, "allc_full.tsv.tbi")):
        continue

    download_binary_file(file_url=url, output_path=output_path)
    shutil.unpack_archive(output_path, extracted_output_path)

    allc_per_chromosome_dfs = []
    per_chromosome_file_list = os.listdir(extracted_output_path)

    for chromosome in tqdm(chromosome_list):
        chromosome_allc_file_name = [x for x in per_chromosome_file_list if x.endswith(f"_{chromosome}.tsv")][0]
        chromosome_allc_file_path = os.path.join(extracted_output_path, chromosome_allc_file_name)
        allc_per_chromosome_dfs.append(
            pd.read_csv(chromosome_allc_file_path, sep="\t")
        )

    allc_full = pd.concat(allc_per_chromosome_dfs)

    output_path = os.path.join(extracted_output_path, "allc_full.tsv")
    allc_full.to_csv(output_path, header=False, sep="\t", index=False)
    standardize_allc(allc_path=output_path, chrom_size_path=chrom_size_path)

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
allc_table_path = "/projects/leelab2/metVI/Mo_2015_MethylC-seq/allc_table.tsv"
allc_table.to_csv(allc_table_path, header=False, sep="\t")

# Download and preprocess gene body annotation file for use by ALLCools.
# AWK command taken from endrebak's answer at https://www.biostars.org/p/56280/
gene_body_annotation_url = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M22/gencode.vM22.annotation.gtf.gz"
download_binary_file(
    gene_body_annotation_url,
    "/projects/leelab2/metVI/Mo_2015_MethylC-seq/gencode.vM22.annotation.gtf.gz",
)
os.chdir("/projects/leelab2/metVI/Mo_2015_MethylC-seq/")
os.system(
    'zcat gencode.vM22.annotation.gtf.gz | awk \'OFS="\t" {if ($3=="gene") {print $1,$4-1,$5,$10,$16,$7}}\' | tr -d \'";\' > mm10_gene_bodies.bed'
)

# Create MCDS file with all desired regions
mcds_output_path = "/projects/leelab2/metVI/Mo_2015_MethylC-seq/Mo_2015_MethylC-seq.mcds"
allc_table_path = "/projects/leelab2/metVI/Mo_2015_MethylC-seq/allc_table.tsv"
os.chdir("/projects/leelab2/metVI/Mo_2015_MethylC-seq/")

# Create MCDS file with all desired regions
if os.path.exists(mcds_output_path):
    print("MCDS file already present. Skipping MCDS file generation...")
else:
    print("Generating MCDS file")
    os.system(
        f"allcools generate-dataset \
        --allc_table allc_table.tsv \
        --output_path Mo_2015_MethylC-seq.mcds \
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

    metadata = pd.DataFrame(index=mcds.cell.indexes['cell'])
    metadata['cell_type'] = [x.split("_")[0] for x in metadata.index]

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