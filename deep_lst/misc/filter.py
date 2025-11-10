import numpy as np
import pandas as pd

class SpatialFilter:

    def __init__(self, store, spatial_indices):

        self.store = store
        self.spatial_indices = spatial_indices
        self.P = len(self.spatial_indices)

    def _land_mask(self):

        if "LWMASK" in self.store:
            mask = self.store["LWMASK"]
        elif "lsm" in self.store:
            mask = self.store["lsm"]
        else:
            raise ValueError("No land water mask found in store.")

        # Drop time if present
        if "time" in mask.dims:
            mask = mask.isel(time=0)
        
        self.land_mask = mask.values[self.spatial_indices["lat_idx"].to_numpy(), self.spatial_indices["lon_idx"].to_numpy()] == 1 # land should be 1, inland water 3

        return self.land_mask  # shape (lat, lon)


    def apply_mask(self, masking = "land"):

        if masking == "land":
            mask = self._land_mask()

        self.spatial_indices = self.spatial_indices.loc[mask].reset_index(drop=True)

        return self.spatial_indices
    
    def apply_rgn(self, frac):
        return self.spatial_indices.sample(frac=frac, random_state=42).reset_index(drop=True)
    

class TemporalFilter:

    def __init__(self, store, time_index):

        self.store = store
        self.time_index = time_index
        self.T = len(self.time_index)

    def apply_mask(self, start_date, end_date):
        mask = (self.time_index >= start_date) & (self.time_index < end_date)
        self.temporal_indices = np.nonzero(mask)[0].astype(np.int64)
        return self.temporal_indices

    def apply_rgn(self, frac, seed=42):
        k = max(1, int(round(len(self.temporal_indices) * frac))) # ensures at least one sample
        rng = np.random.default_rng(seed)
        self.temporal_indices = np.sort(rng.choice(self.temporal_indices, size=k, replace=False))
        return self.temporal_indices

    def apply_stratification(self, frac, by: str = "month", seed=42):
        
        """
        Returns a 1D np.array of int time indices sampled without replacement,
        stratified by time. 
        """
                
        # build stratification dataframe
        timeidx_subset = self.time_index[self.temporal_indices]
        df = pd.DataFrame({"i": self.temporal_indices, "t": timeidx_subset})
        df["month"] = df["t"].dt.month              # 1..12
        df["hour"]  = df["t"].dt.hour

        if by == "month":
            df["stratum"] = df["month"]
        elif by == "hour":
            df["stratum"] = df["hour"]
        elif by == "month_hour":
            df["stratum"] = list(zip(df["month"], df["hour"]))
        else:
            raise ValueError("by must be 'month', 'hour', or 'month_hour'")

        # How many total to draw?
        total_samples = max(1, int(round(len(self.temporal_indices) * frac)))
        # Compute per-stratum allocations
        grouped_indices = df.groupby("stratum", sort=False)
        stratified_samples = grouped_indices.size().rename("stratified_samples")

         # proportional to size with rounding that preserves the total
        props = stratified_samples / stratified_samples.sum()
        raw = (props * total_samples).to_numpy()
        flo = np.floor(raw).astype(int)
        rem = total_samples - flo.sum()
        # give the remaining items to strata with largest fractional parts
        fractional_order = np.argsort(raw - flo)[::-1][:rem]
        stratification = pd.Series(flo, index=stratified_samples.index)
        stratification.iloc[fractional_order] += 1
        stratification = stratification.clip(upper=stratified_samples)

        rng = np.random.default_rng(seed)
        samples_to_take = []
        for stratum, samples in stratification.items():
            if samples <= 0:
                continue
            stratum_indices = grouped_indices.get_group(stratum)["i"].to_numpy()
            # If k == len(pool), choice without replacement returns a permutation
            samples_to_take.append(rng.choice(stratum_indices, size=int(samples), replace=False))

        # Concatenate and (optionally) shuffle so DataLoader batches mix strata
        time_indices = np.concatenate(samples_to_take)
        rng.shuffle(time_indices)
        self.temporal_indices = time_indices
        return self.temporal_indices