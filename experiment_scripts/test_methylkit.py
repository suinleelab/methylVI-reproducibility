import mudata
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import globalenv
from rpy2.robjects import numpy2ri
from rpy2.robjects import pandas2ri

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

groups = (
    ("Excitatory", "PV"),
    ("PV", "VIP"),
    ("Excitatory", "VIP"),
)


def get_neuron_type(x):
    if x in excitatory_subtypes:
        return "Excitatory"
    elif x == "mVip":
        return "VIP"
    elif x == "mPv":
        return "PV"
    else:
        return "Other"

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

for modality in ['mCH', 'mCG']:
    adata = mdata_single_cell[modality]
    adata.obs['Coarse_type'] = [get_neuron_type(x) for x in adata.obs["Neuron type"]]

    for (group1, group2) in groups:

        adata_ = adata[adata.obs['Coarse_type'].isin([group1, group2])]

        var_names = adata_.var_names
        cell_types = adata_.obs['Coarse_type']

        coverage = adata_.layers['cov']
        num_c = adata_.layers['mc']
        num_t = coverage - num_c

        full_data = np.dstack(
            (coverage.T, num_c.T, num_t.T)
        ).reshape(coverage.shape[1], -1).astype(np.int32)

        coverage_idx = np.arange(0, coverage.shape[0]) * 3 + 1 + 4
        num_c_idx = coverage_idx + 1
        num_t_idx = num_c_idx + 1

        coverage_idx_names = [f"coverage {x}" for x in np.arange(coverage.shape[0])]
        num_c_idx_names = [f"numCs {x}" for x in np.arange(coverage.shape[0])]
        num_t_idx_names = [f"numTs {x}" for x in np.arange(coverage.shape[0])]

        sample_ids = adata_.obs.index

        context = "CpH"
        resolution = "region"
        destranded = False
        treatment = [1 if x == group1 else 2 for x in cell_types]

        globalenv['num_t_idx'] = ro.IntVector(num_t_idx)
        globalenv['num_c_idx'] = ro.IntVector(num_c_idx)
        globalenv['coverage_idx'] = ro.IntVector(coverage_idx)

        globalenv['num_t_idx_names'] = ro.StrVector(num_t_idx_names)
        globalenv['num_c_idx_names'] = ro.StrVector(num_c_idx_names)
        globalenv['coverage_idx_names'] = ro.StrVector(coverage_idx_names)

        globalenv['treatment'] = ro.IntVector(treatment)

        globalenv['sample_ids'] = sample_ids
        globalenv['full_data'] = full_data
        globalenv['context'] = context
        globalenv['var_names'] = adata_.var_names

        ro.r(
        f'''
        library(methylKit)
        
        df <- as.data.frame(full_data)
        df <- cbind(chr=var_names, start=42, end=42, strand="*", df)
        
        print(length(coverage_idx))
        print(length(coverage_idx_names))
        
        names(df)[coverage_idx] <- coverage_idx_names
        names(df)[num_t_idx] <- num_t_idx_names
        names(df)[num_c_idx] <- num_c_idx_names
        
        obj <- new(
            "methylBase",
            df,
            sample.ids=sample_ids,
            assembly="hg18", # Dummy placeholder
            context=context, # Dummy placeholder
            treatment=treatment,
            coverage.index=coverage_idx,
            numCs.index=num_c_idx,
            numTs.index=num_t_idx,
            destranded=FALSE,
            resolution="region"
        )
        
        pooled_obj <- pool(obj, sample.ids=c("control", "treatment"))
        myDiff <- calculateDiffMeth(pooled_obj, mc.cores=1)
        write.csv(cbind(chr=myDiff$chr, qvalue=myDiff$qvalue, meth.diff=myDiff$meth.diff), "methylkit_{group1}_{group2}_{modality}_{n_features}.csv")
        '''
        )
