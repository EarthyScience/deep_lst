"""Contract of the store readers.

A reader gives access to one store. Method and attribute names follow the eodata readers that
these classes replace; what they return is plain numpy.
"""

from abc import ABC, abstractmethod


class StoreReader(ABC):
    """What every reader in deep_lst.stores provides.

    Attributes:
        modality_id: Name of the store's content, e.g. "seviri_product_hourly_lst".
        variables: Names of the variables. Their order is the order of the last axis of
            everything the reader returns.
        spatial_grid: Grid of the store; a spatial index is a row of this grid.
        time_grid: Time axis of the store; a time index is a position on it. None for a
            static store, whose only time index is 0.
    """

    modality_id: str
    variables: list[str]

    @property
    def is_static(self):
        return self.time_grid is None

    @property
    @abstractmethod
    def available_indices(self):
        """(spatial index, time index) of everything that was written to the store. (n, 2)"""

    @abstractmethod
    def get_data_single(self, spatial_idx, time_idx):
        """Values of one cell at one time step.

        Returns:
            float32 array (n_variables,), NaN where the store has no value.
        """

    @abstractmethod
    def get_data_multi(self, indices):
        """Values for several (spatial index, time index) pairs.

        Args:
            indices: Pairs (spatial_idx, time_idx), as a list or an (n, 2) array.

        Returns:
            float32 array (n, n_variables) in the order of the pairs, NaN where the store has
            no value.
        """
