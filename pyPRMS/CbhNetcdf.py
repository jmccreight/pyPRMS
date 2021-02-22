import datetime
import pandas as pd
import netCDF4 as nc
import xarray as xr

from typing import List, Optional

# from typing import Any,  Union, Dict, List, OrderedDict as OrderedDictType

CBH_VARNAMES = ['prcp', 'tmin', 'tmax']
CBH_INDEX_COLS = [0, 1, 2, 3, 4, 5]


class CbhNetcdf(object):
    """Climate-By-HRU (CBH) files in netCDF format"""

    # Author: Parker Norton (pnorton@usgs.gov)
    # Create date: 2019-04-30
    # Description: Class for working with individual cbh files

    # 2016-12-20 PAN:
    # As written this works with CBH files that were created with
    # java class gov.usgs.mows.GCMtoPRMS.GDPtoCBH
    # This program creates the CBH files in the format needed by PRMS
    # and also verifies the correctness of the data including:
    #    tmax is never less than tmin
    #    prcp is never negative
    #    any missing data/missing date is filled with (?? avg of bracketing dates??)
    #
    # I think it might be better if this code worked with the original GDP files and
    # took care of those corrections itself. This would provide a more seamless workflow
    # from GDP to PRMS. At this point I'm not taking this on though -- for a future revision.

    def __init__(self, src_path: Optional[str] = None,
                 st_date: Optional[datetime.datetime] = None,
                 en_date: Optional[datetime.datetime] = None,
                 nhm_hrus: Optional[List[int]] = None):
        """
        :param src_path: Full path to netCDF file
        :param st_date: The starting date for restricting CBH results
        :param en_date: The ending date for restricting CBH results
        :param nhm_hrus: List of NHM HRU IDs to extract from CBH
        """
        self.__src_path = src_path
        self.__stdate = st_date
        self.__endate = en_date
        self.__nhm_hrus = nhm_hrus
        self.__date_range = None
        self.__dataset = None
        self.__final_outorder = None

        self.read_netcdf()

    def read_netcdf(self):
        """Read CBH files stored in netCDF format
        """

        if self.__nhm_hrus:
            # print('\t\tOpen dataset')
            try:
                self.__dataset = xr.open_mfdataset(self.__src_path,
                                                   chunks={'hruid': 1040}, combine='by_coords',
                                                   decode_cf=True)
                # NOTE: With a multi-file dataset the time attributes 'units' and
                #       'calendar' are lost.
                #       see https://github.com/pydata/xarray/issues/2436
            except ValueError:
                self.__dataset = xr.open_mfdataset(self.__src_path, chunks={'hru': 1040}, combine='by_coords')
        else:
            print('ERROR: write the code for all HRUs')
            exit()

    def get_var(self, var: str) -> pd.DataFrame:
        """
        Get a variable from the netCDF file

        :param var: Name of the variable

        :return: Data for the variable
        """
        if self.__stdate is not None and self.__endate is not None:
            # print(var, type(var))
            # print(self.__stdate, type(self.__stdate))
            # print(self.__endate, type(self.__endate))
            # print(self.__nhm_hrus, type(self.__nhm_hrus))
            try:
                data = self.__dataset[var].loc[self.__stdate:self.__endate, self.__nhm_hrus].to_pandas()
            except IndexError:
                print(f'ERROR: Indices (time, hruid) were used to subset {var} which expects' +
                      f'indices ({" ".join(map(str, self.__dataset[var].coords))})')
                raise
        else:
            data = self.__dataset[var].loc[:, self.__nhm_hrus].to_pandas()

        return data

    def write_ascii(self, pathname: Optional[str] = None,
                    fileprefix: Optional[str] = None,
                    variables: Optional[List[str]] = None):
        """Write CBH data for variables to ASCII formatted file(s).

        :param pathname: Path to write files
        :param fileprefix: Filename prefix
        :param variables: CBH variable(s) to write to file(s). If None then all variables in netCDF file are output.
        """
        # For out_order the first six columns contain the time information and
        # are always output for the cbh files
        out_order = [kk for kk in self.__nhm_hrus]
        for cc in ['second', 'minute', 'hour', 'day', 'month', 'year']:
            out_order.insert(0, cc)

        var_list = []
        if variables is None:
            var_list = self.__dataset.data_vars
        elif isinstance(variables, list):
            var_list = variables

        for cvar in var_list:
            if cvar in self.__dataset.data_vars:
                data = self.get_var(var=cvar)

                # Add time information as columns
                data['year'] = data.index.year
                data['month'] = data.index.month
                data['day'] = data.index.day
                data['hour'] = 0
                data['minute'] = 0
                data['second'] = 0

                # Output ASCII CBH files
                if fileprefix is None:
                    outfile = f'{cvar}.cbh'
                else:
                    outfile = f'{fileprefix}_{cvar}.cbh'

                if pathname is not None:
                    outfile = f'{pathname}/{outfile}'

                out_cbh = open(outfile, 'w')
                out_cbh.write('Written by Bandit\n')
                out_cbh.write(f'{cvar} {len(self.__nhm_hrus)}\n')
                out_cbh.write('########################################\n')
                # data.to_csv(out_cbh, columns=out_order, na_rep='-999', float_format='%0.3f',
                data.to_csv(out_cbh, columns=out_order, na_rep='-999', float_format='%0.2f',
                            sep=' ', index=False, header=False, encoding=None, chunksize=50)
                out_cbh.close()
            else:
                print(f'WARNING: {cvar} does not exist in source CBH files..skipping')

    def write_netcdf_old(self, filename: str = None,
                         variables: Optional[List[str]] = None):
        """Write CBH to netcdf format file
        """

        # NetCDF-related variables
        var_desc = {'tmax': 'Maximum Temperature', 'tmin': 'Minimum temperature',
                    'prcp': 'Precipitation', 'rhavg': 'Mean relative humidity'}
        var_units = {'tmax': 'C', 'tmin': 'C', 'prcp': 'inches', 'rhavg': 'percent'}

        # Create a netCDF file for the CBH data
        nco = nc.Dataset(filename, 'w', clobber=True)
        # nco.createDimension('hru', len(self.__dataset['hru'].loc[self.__nhm_hrus]))
        nco.createDimension('hru', len(self.__nhm_hrus))
        nco.createDimension('time', None)

        timeo = nco.createVariable('time', 'f4', 'time')
        timeo.calendar = 'standard'
        # timeo.bounds = 'time_bnds'

        # FIXME: Days since needs to be set to the starting date of the model pull
        timeo.units = 'days since 1980-01-01 00:00:00'

        hruo = nco.createVariable('hru', 'i4', 'hru')
        hruo.long_name = 'Hydrologic Response Unit ID (HRU)'

        var_list = []
        if variables is None:
            var_list = self.__dataset.data_vars
        elif isinstance(list, variables):
            var_list = variables

        for cvar in var_list:
            varo = nco.createVariable(cvar, 'f4', ('time', 'hru'), fill_value=nc.default_fillvals['f4'], zlib=True)
            varo.long_name = var_desc[cvar]
            varo.units = var_units[cvar]

        nco.setncattr('Description', 'Climate by HRU')
        # nco.setncattr('Bandit_version', __version__)
        # nco.setncattr('NHM_version', nhmparamdb_revision)

        # Write the HRU ids
        hruo[:] = self.__dataset['hru'].loc[self.__nhm_hrus].values
        hruo[:] = self.__nhm_hrus
        timeo[:] = nc.date2num(pd.to_datetime(self.__dataset['time'].loc[self.__stdate:self.__endate].values).tolist(),
                               units='days since 1980-01-01 00:00:00',
                               calendar='standard')

        for cvar in var_list:
            data = self.get_var(var=cvar)

            # Write the CBH values
            nco.variables[cvar][:, :] = data.values

        nco.close()

    def write_netcdf(self, filename: str = None,
                     variables: Optional[List[str]] = None):
        """Write CBH to netcdf format file

        :param filename: NetCDF filename to write
        :param variables: List of CBH variables to write
        """

        # Create a netCDF file for the CBH data
        nco = nc.Dataset(filename, 'w', clobber=True)
        # nco.createDimension('hru', len(self.__dataset['hru'].loc[self.__nhm_hrus]))
        nco.createDimension('hruid', len(self.__nhm_hrus))
        nco.createDimension('time', None)

        reference_time = self.__stdate.strftime('%Y-%m-%d %H:%M:%S')
        cal_type = 'standard'

        # Create the variables
        timeo = nco.createVariable('time', 'f4', 'time')
        timeo.long_name = 'time'
        timeo.standard_name = 'time'
        timeo.calendar = cal_type
        timeo.units = f'days since {reference_time}'

        hruo = nco.createVariable('hruid', 'i4', 'hruid')
        hruo.long_name = 'Hydrologic Response Unit ID (HRU)'
        hruo.cf_role = 'timeseries_id'

        var_list = []
        if variables is None:
            var_list = self.__dataset.data_vars
        elif isinstance(variables, list):
            var_list = variables

        for cvar in var_list:
            cxry = self.__dataset[cvar]
            varo = nco.createVariable(cvar, cxry.encoding['dtype'], cxry.dims,
                                      fill_value=cxry.attrs['fill_value'],
                                      zlib=True)
            varo.long_name = cxry.attrs['long_name']
            varo.units = cxry.attrs['units']

            if 'standard_name' in cxry.attrs:
                varo.standard_name = cxry.attrs['standard_name']

        nco.setncattr('Description', 'Climate by HRU')
        # nco.setncattr('Bandit_version', __version__)
        # nco.setncattr('NHM_version', nhmparamdb_revision)

        # Write the HRU ids
        hruo[:] = self.__dataset['hruid'].loc[self.__nhm_hrus].values
        hruo[:] = self.__nhm_hrus

        # Write time information
        timeo[:] = nc.date2num(pd.to_datetime(self.__dataset['time'].loc[self.__stdate:self.__endate].values).tolist(),
                               units=f'days since {reference_time}',
                               calendar=cal_type)

        for cvar in var_list:
            data = self.get_var(var=cvar)

            # Write the CBH values
            nco.variables[cvar][:, :] = data.values

        nco.close()
