import mudata
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import globalenv
from rpy2.robjects import numpy2ri
from rpy2.robjects import pandas2ri
from rpy2.robjects.vectors import StrVector

numpy2ri.activate()
pandas2ri.activate()

n_features = "2500"

mdata_single_cell = mudata.read_h5mu(
    f"/projects/leelab2/metVI/Luo2017_Mouse/data/gene_{n_features}_features.h5mu"
)

excitatory_subtypes = [
    "mL2/3",
    "mL4",
    "mL5-1",
    "mDL-1",
    "mDL-2",
    "mDL-3",
    "mIn-1",
    "mL6-1",
    "mL6-2",
    "mL5-2"
]

def get_neuron_type(x):
    if x in excitatory_subtypes:
        return "Excitatory"
    elif x == "mVip":
        return "VIP"
    elif x == "mPv":
        return "PV"
    else:
        return "Other"

groups = (
    ("Excitatory", "PV"),
    ("PV", "VIP"),
    ("Excitatory", "VIP"),
)

def get_neuron_type(x):
    if x in excitatory_subtypes:
        return "Excitatory"
    elif x == "mVip":
        return "Inhibitory"
    elif x == "mPv":
        return "Inhibitory"
    else:
        return "Other"

groups = (
    ("Excitatory", "Inhibitory"),
)

for modality in ['mCG', 'mCH']:
    adata = mdata_single_cell[modality]
    adata.obs['Coarse_type'] = [get_neuron_type(x) for x in adata.obs["Neuron type"]]

    for (group1, group2) in groups:
        adata_ = adata[adata.obs['Coarse_type'].isin([group1, group2])]

        var_names = adata_.var_names
        cell_types = adata_.obs['Coarse_type']

        sample_labels = [f"{x}_{i}" for i, x in enumerate(cell_types)]

        group1_sample_labels = [x for x in sample_labels if x.startswith(group1)]
        group2_sample_labels = [x for x in sample_labels if x.startswith(group2)]

        globalenv['var_names'] = StrVector(var_names)
        globalenv['sample_labels'] = StrVector(sample_labels)
        globalenv['group1_sample_labels'] = StrVector(group1_sample_labels)
        globalenv['group2_sample_labels'] = StrVector(group2_sample_labels)

        dss_formatted_data = [np.stack([
            adata_.layers['cov'][i, :],
            adata_.layers['mc'][i, :]
        ]).T for i in range(adata_.shape[0])]
        dss_formatted_data = np.stack(dss_formatted_data).astype(np.int32)
        globalenv['dss_formatted_data'] = dss_formatted_data

        ro.r(
        f'''
            library(DSS)
            require(bsseq)
            
            create_df = function(i) {{
                N = dss_formatted_data[i,, 1]
                X = dss_formatted_data[i,, 2]
                chr = var_names
                pos = rep(42, length(chr))
                
                df = data.frame(chr, pos, N, X)
                return(df)
            }}
            
            df_list = lapply(1:dim(dss_formatted_data)[1], create_df)
            
            start = Sys.time()
            BSobj = makeBSseqData(
                df_list,
                sample_labels
            )
            end = Sys.time()
            print(end - start)
            
            start = Sys.time()
            dmlTest = DMLtest(
                BSobj,
                group1=group1_sample_labels,
                group2=group2_sample_labels,
                ncores=40,
            )
            
            write.csv(dmlTest, "dss_{group1}_{group2}_{modality}_{n_features}.csv")
            end = Sys.time()
            print(end - start)
        '''
        )