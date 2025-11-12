import xarray as xr
import numpy as np
import pandas as pd

from deep_lst.mlp.datareader import weTestDataset
from deep_lst.misc.filter import SpatialFilter

def reshape_results(inference_results, TSTEPS):
    n_repeat = len(inference_results) // TSTEPS 
    t_indices = np.arange(TSTEPS)
    inference_results["timestep"] = np.repeat(t_indices, n_repeat)
    return inference_results

def results_store(config, seviri_indices):

    testData = weTestDataset(
        config=config,
        start_time=config['test_start'],
        end_time=config['test_end'],
    )

    store = testData.reader_seviri.store
    filter = SpatialFilter(store, seviri_indices)
    land_mask = filter._land_mask()   

    df = seviri_indices.copy()
    df["targ"] = pd.Series(np.full(len(land_mask), np.nan), index=df.index, dtype="float64")
    df["pred"] = pd.Series(np.full(len(land_mask), np.nan), index=df.index, dtype="float64")

    return df, land_mask

def assemble_dataarray(df, values):

    tmp = (df.assign(src_lat_r=df['src_lat'].round(5),
                     src_lon_r=df['src_lon'].round(5))
             .groupby(['src_lat_r','src_lon_r'], as_index=False)[values].first()
             .pivot(index='src_lat_r', columns='src_lon_r', values=values)) # ragged gridd
    da = xr.DataArray(tmp.to_numpy(),
                      dims=('latitude','longitude'),
                      coords={'latitude': tmp.index.to_numpy(),
                              'longitude': tmp.columns.to_numpy()})
    return da