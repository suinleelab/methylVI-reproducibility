import os
import re
from collections import defaultdict

import biomart
import pandas as pd
import requests
import scanpy as sc
from ALLCools.clustering import log_scale


def download_binary_file(
    file_url: str, output_path: str, overwrite: bool = False
) -> None:
    """
    Download binary data file from a URL.

    Args:
    ----
        file_url: URL where the file is hosted.
        output_path: Output path for the downloaded file.
        overwrite: Whether to overwrite existing downloaded file.

    Returns
    -------
        None.
    """
    file_exists = os.path.exists(output_path)
    if (not file_exists) or (file_exists and overwrite):
        request = requests.get(file_url)
        with open(output_path, "wb") as f:
            f.write(request.content)
        print(f"Downloaded data from {file_url} at {output_path}")
    else:
        print(
            f"File {output_path} already exists. "
            "No files downloaded to overwrite the existing file."
        )
        
        
        
# These next two function were lightly modified from those found in the
# ALLCools repository
def make_obs_df_var_df(use_data, obs_dim, var_dim):
    obs_df = pd.DataFrame([], index=use_data.get_index(obs_dim).astype(str))
    var_df = pd.DataFrame([], index=use_data.get_index(var_dim).astype(str))
    coord_prefix = re.compile(f"({obs_dim}|{var_dim})_")
    for k, v in use_data.coords.items():
        if k in [obs_dim, var_dim]:
            continue
        try:
            # v.dims should be size 1
            if v.dims[0] == obs_dim:
                series = v.to_pandas()
                # adata.obs_name is str type
                series.index = series.index.astype(str)
                obs_df[coord_prefix.sub("", k)] = series
            elif v.dims[0] == var_dim:
                series = v.to_pandas()
                # adata.var_name is str type
                series.index = series.index.astype(str)
                var_df[coord_prefix.sub("", k)] = series
            else:
                pass
        except IndexError:
            # v.dims is 0, just ignore
            pass
    return obs_df, var_df



def mcds_to_anndata(mcds, mc_pattern, var_dim, obs_dim, metadata, subset_features):
    if subset_features:
        use_features = (
            mcds.coords[f"{var_dim}_{mc_pattern}_feature_select"]
                .to_pandas()
                .dropna()
                .astype(bool)
        )
        use_features = use_features[use_features].index

        cov = mcds[f'{var_dim}_da'].sel({"mc_type": mc_pattern, var_dim: use_features}).sel({'count_type': 'cov'}).squeeze()
        mc = mcds[f'{var_dim}_da'].sel({"mc_type": mc_pattern, var_dim: use_features}).sel({'count_type': 'mc'}).squeeze()
        mc_frac = mcds[f'{var_dim}_da_frac'].sel({"mc_type": mc_pattern, var_dim: use_features}).squeeze()
    else:
        cov = mcds[f'{var_dim}_da'].sel({"mc_type": mc_pattern}).sel({'count_type': 'cov'}).squeeze()
        mc = mcds[f'{var_dim}_da'].sel({"mc_type": mc_pattern}).sel({'count_type': 'mc'}).squeeze()
        mc_frac = mcds[f'{var_dim}_da_frac'].sel({"mc_type": mc_pattern}).squeeze()
    obs_df, var_df = make_obs_df_var_df(mc_frac, obs_dim, var_dim)

    adata = sc.AnnData(
        X=mc_frac.values,
        var=var_df,
        obs=obs_df
    )
    adata.obs = metadata.loc[adata.obs.index]
    adata.layers['cov'] = cov.values
    adata.layers['mc'] = mc.values
    log_scale(adata)
    return adata

def get_mm_ensembl_mappings():
    # Set up connection to server
    server = biomart.BiomartServer('http://ensembl.org/biomart')
    mart = server.datasets['mmusculus_gene_ensembl']

    # List the types of data we want
    attributes = ['ensembl_transcript_id', 'mgi_symbol',
                  'ensembl_gene_id', 'ensembl_peptide_id']

    # Get the mapping between the attributes
    response = mart.search({'attributes': attributes})
    data = response.raw.data.decode('ascii')

    ensembl_to_genesymbol = defaultdict(str)
    # Store the data in a dict
    for line in data.splitlines():
        line = line.split('\t')
        # The entries are in the same order as in the `attributes` variable
        transcript_id = line[0]
        gene_symbol = line[1]
        ensembl_gene = line[2]
        ensembl_peptide = line[3]

        # Some of these keys may be an empty string. If you want, you can
        # avoid having a '' key in your dict by ensuring the
        # transcript/gene/peptide ids have a nonzero length before
        # adding them to the dict
        ensembl_to_genesymbol[transcript_id] = gene_symbol
        ensembl_to_genesymbol[ensembl_gene] = gene_symbol
        ensembl_to_genesymbol[ensembl_peptide] = gene_symbol

    return ensembl_to_genesymbol


def get_hs_ensembl_mappings():
    # Set up connection to server
    server = biomart.BiomartServer('http://ensembl.org/biomart')
    mart = server.datasets['hsapiens_gene_ensembl']

    # List the types of data we want
    attributes = [
        'ensembl_transcript_id',
        'hgnc_symbol',
        'ensembl_gene_id',
        'ensembl_peptide_id',
        'transcript_biotype',
        'start_position',
        'end_position'
    ]

    # Get the mapping between the attributes
    response = mart.search({'attributes': attributes})
    data = response.raw.data.decode('ascii')

    results = defaultdict(lambda: defaultdict(str))

    # Store the data in a dict
    for line in data.splitlines():
        line = line.split('\t')
        # The entries are in the same order as in the `attributes` variable
        transcript_id = line[0]
        gene_symbol = line[1]
        ensembl_gene = line[2]
        ensembl_peptide = line[3]
        biotype = line[4]
        start_position = line[5]
        end_position = line[6]

        # Some of these keys may be an empty string. If you want, you can
        # avoid having a '' key in your dict by ensuring the
        # transcript/gene/peptide ids have a nonzero length before
        # adding them to the dict
        results[ensembl_gene]["transcript_id"] = transcript_id
        results[ensembl_gene]["gene_symbol"] = gene_symbol
        results[ensembl_gene]["ensembl_peptide"] = ensembl_peptide
        results[ensembl_gene]["biotype"] = biotype
        results[ensembl_gene]["start_position"] = start_position
        results[ensembl_gene]["end_position"] = end_position

    return results