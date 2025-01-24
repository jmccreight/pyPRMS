import numpy as np
import os
import pandas as pd
import xarray as xr

from typing import Union

from ..constants import NEW_PTYPE_TO_DTYPE


class OutputVariable(object):
    """Container for a single output variable
    """

    def __init__(self, name: str, filename: Union[str, os.PathLike], metadata: dict):
        """Initialize the OutputVariable object.

        :param name: Name of the output variable
        :param filename: Path to the model output variable file
        :param metadata: Metadata for the output variable
        """

        self.__name = name
        self.__filename = filename
        self.__data = None

        if 'variables' in metadata:
            self.metadata = metadata['variables'][name]
        else:
            self.metadata = metadata[name]

    @property
    def data(self) -> pd.DataFrame:
        """Returns the source model output.

        :returns: Model output dataframe
        """
        if self.__data is None:
            self._read_file()

        return self.__data

    @property
    def filename(self) -> Union[str, os.PathLike]:
        """Return the path to the model output variable

        :returns: Path to the model output variable file
        """

        return self.__filename

    def to_xarray(self) -> xr.DataArray:
        """Return output variable as an xarray object.

        :returns: xarray DataArray
        """

        local_dim_desc = {'nhru': 'Local model Hydrologic Response Unit ID (HRU)',
                          'nsegment': 'Local model segment ID'}

        global_dims = dict(nhru=dict(varname='nhm_id',
                                     long_name='NHM Hydrologic Response Unit ID (HRU)'),
                           nsegment=dict(varname='nhm_seg',
                                         long_name='NHM segment ID'))

        dim_name = self.metadata['dimensions'][0]

        if dim_name == 'one':
            # Basin variable
            da = self.data.squeeze().to_xarray()
        else:
            # Pivot dataframe, adding index for nhru or nsegment (multi-index)
            df = self.data.stack()

            # Rename the multi-index names (time and either nhru or nsegment)
            df.index.names = [self.data.index.name, dim_name]

            # Name the data-series as the variable name
            df.name = self.__name

            # Convert to xarray and set attributes and encoding
            da = df.to_xarray()

            if self.metadata.get('is_global', False):
                # When is_global is true the file header contains global HRU or segment IDs
                da[global_dims[dim_name]['varname']] = da[dim_name]
                da[global_dims[dim_name]['varname']].attrs['long_name'] = global_dims[dim_name]['long_name']

                # Reset the nhru/nsegment coordinate variable values to 1..N
                da[dim_name] = np.arange(1, self.data.shape[1]+1, dtype=np.int32)

            # Set attributes for local model dimensions
            da[dim_name].attrs['long_name'] = local_dim_desc[dim_name]

        # Set the time coordinate variable attributes
        first_time = self.__data.index[0]
        da.time.attrs['standard_name'] = 'time'
        da.time.attrs['long_name'] = 'time'
        da.time.encoding['units'] = f'days since {first_time.year}-{first_time.month:02d}-{first_time.day:02d} 00:00:00'
        da.time.encoding['calendar'] = 'standard'

        # Output variable attributes
        da.attrs['long_name'] = self.metadata['description']
        da.attrs['units'] = self.metadata['units']
        da.encoding.update(dict(_FillValue=None,
                                compression='zlib',
                                complevel=2,
                                fletcher32=True))

        return da

    def _read_file(self):
        """Read model variable output file.
        """

        self.__data = pd.read_csv(self.__filename, sep=',', skipinitialspace=True,
                                  header=0, index_col=0, parse_dates=True,
                                  dtype=NEW_PTYPE_TO_DTYPE[self.metadata['datatype']])

        self.__data.index.name = 'time'

        if self.metadata['dimensions'][0] in ['nhru', 'nsegment', 'nsub']:
            self.__data.columns = self.__data.columns.astype(np.int32)
        elif self.metadata['dimensions'][0] == 'one':
            self.__data = self.__data.loc[:, self.__name].to_frame()
