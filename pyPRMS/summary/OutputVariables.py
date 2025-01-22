
import os
os.environ['USE_PYGEOS'] = '0'
import geopandas   # type: ignore

import cartopy.crs as ccrs  # type: ignore
import netCDF4
import pandas as pd   # type: ignore

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER  # type: ignore
from functools import cached_property
from typing import Dict, List, Optional, Tuple, Union

import matplotlib as mpl        # type: ignore
import matplotlib.pyplot as plt     # type: ignore

from .OutputVariable import OutputVariable
from ..plot_helpers import (set_colormap, get_projection, plot_line_collection, plot_polygon_collection,
                            get_figsize, read_gis)
from ..prms_helpers import chunk_shape_2d
from ..constants import NETCDF_DATATYPES, PTYPE_TO_PRMS_TYPE

# - pass control object
# - list available variables
#     - basin (basinON_OFF = 1)
#     - streamflow (csvON_OFF = 2) 
#     - segment (nsegmentON_OFF = 1,2)
#     - hru (nhruON_OFF = 1,2)
#     - <others>
# - get variable


class OutputVariables(object):
    def __init__(self,
                 control,
                 metadata,
                 model_dir=None,
                 verbose: Optional[bool] = False):
        """Initialize the model output object.
        """

        self.__control = control
        self.metadata = metadata['variables']
        self.__dimension_metadata = metadata['dimensions']
        self.__model_dir = model_dir
        self.__verbose = verbose

        self.__out_vars = dict()

        self.__hru_poly = None
        self.__hru_shape_key: Optional[str] = None
        self.__seg_poly = None
        self.__seg_shape_key: Optional[str] = None

        self.__use_global = dict(nhru=False, nsegment=False, nsub=False)

        for cvar, cfile in self.available_vars.items():
            self.__out_vars[cvar] = OutputVariable(cvar, cfile, self.metadata)

        print(f'global IDS: {self.__use_global}')

    @cached_property
    def available_vars(self):
        """Returns dictionary of available variables and file paths"""
        var_dict = {}

        var_kind = dict(nhru=[1, 2],
                        nsegment=[1, 2],
                        nsub=[1])

        for ckind, cvals in var_kind.items():
            if self.__control.get(f'{ckind}OutON_OFF').values in cvals:
                self.__use_global[ckind] = self.__control.get(f'{ckind}OutON_OFF').values == 2

                prefix = self.__control.get(f'{ckind}OutBaseFileName').values
                varlist = self.__control.get(f'{ckind}OutVar_names').values.tolist()

                if self.__model_dir:
                    prefix = f'{self.__model_dir}/{prefix}'

                for vv in varlist:
                    var_dict[vv] = f'{prefix}{vv}.csv'

        if self.__control.get('basinOutON_OFF').values == 1:
            filename = self.__control.get('basinOutBaseFileName').values
            varlist = self.__control.get('basinOutVar_names').values.tolist()

            if self.__model_dir:
                filename = f'{self.__model_dir}/{filename}'

            for vv in varlist:
                var_dict[vv] = f'{filename}.csv'

        # if self.__control.get('csvON_OFF').values == 2:
        #     filename = self.__control.get('csv_output_file').values
        #
        #     if self.__model_dir:
        #         filename = f'{self.__model_dir}/{filename}'
        #
        #     var_dict['model_streamflow'] = f'{filename}'

        return var_dict

    def get(self, varname):
        return self.__out_vars[varname]

    def write_netcdf(self,
                     filename: Union[str, os.PathLike],
                     var_name: str,
                     chunks: Optional[Dict[str, int]] = None) -> None:
        """Write model output to netCDF file
        """

        # TODO: 2025-01-22 PAN - this does not work for basin variables

        # Map dimension name to name used in netcdf file
        dim_name_nc = dict(ngw='nhru', nssr='nhru', nhru='nhru', nsegment='nsegment')

        global_dim_var = {'nhru': 'nhm_id',
                          'nsegment': 'nhm_seg'}

        global_dim_desc = {'nhru': 'NHM Hydrologic Response Unit ID (HRU)',
                           'nsegment': 'NHM segment ID'}

        dim_desc = {'nhru': 'Local model Hydrologic Response Unit ID (HRU)',
                    'nsegment': 'Local model segment ID'}

        var_obj = self.get(var_name)
        var_data = var_obj.data

        # Get the mapped dimension name (e.g. ngw, nssr, nhru => nhru)
        curr_dim = dim_name_nc[var_obj.metadata['dimensions'][0]]

        # Get starting date
        st_date = var_data.index[0]   # src_df.index[0]   # var_dates[0]
        # en_date = var_data.index[-1]   # src_df.index[-1]   # var_dates[-1]

        # ntime = (en_date - st_date).days + 1
        ntime = var_data.shape[0]
        ncol = var_data.shape[1]

        if chunks is None:
            cnk_sizes = chunk_shape_2d((ntime, ncol), val_size=4, chunk_size=1048576)  # 32768)
        else:
            cnk_sizes = [chunks[vv] for vv in ['time', curr_dim]]

        nco = netCDF4.Dataset(filename, 'w', clobber=True)

        nco.createDimension(curr_dim, ncol)
        nco.createDimension('time', None)

        timeo = nco.createVariable('time', 'f4', 'time')
        timeo.calendar = 'standard'
        timeo.units = f'days since {st_date.year}-{st_date.month:02d}-{st_date.day:02d} 00:00:00'

        dimo = nco.createVariable(curr_dim, 'i4', curr_dim)
        dimo.long_name = dim_desc[curr_dim]

        var_type = NETCDF_DATATYPES[PTYPE_TO_PRMS_TYPE[var_obj.metadata['datatype']]]
        varo = nco.createVariable(var_name, var_type, ('time', curr_dim),
                                  fill_value=netCDF4.default_fillvals[var_type],
                                  zlib=True, complevel=1,
                                  chunksizes=cnk_sizes)

        varo.long_name = var_obj.metadata['description']
        varo.units = var_obj.metadata['units']

        # Add the global/NHM IDs
        if self.__use_global[curr_dim] and curr_dim in global_dim_var:
            globalido = nco.createVariable(global_dim_var[curr_dim], 'i4', curr_dim)
            globalido.long_name = global_dim_desc[curr_dim]

            globalido[:] = var_data.columns

        # Write the local model HRU or segment IDs
        dimo[:] = list(range(1, ncol+1))

        timeo[:] = netCDF4.date2num(var_data.index.to_pydatetime(), units=timeo.units, calendar=timeo.calendar)

        nco.variables[var_name][:, :] = var_data.values
        # nco.variables[var_name][:, :] = var_data.transpose()

        nco.close()

    # @staticmethod
    # def _read_streamflow_header(filename):
    #     """Read the headers from a PRMS CSV model output file (ON_OFF=2)"""
    #     fhdl = open(filename, 'r')
    #
    #     # First and second rows are headers
    #     hdr1 = fhdl.readline().strip()
    #
    #     fhdl.close()
    #
    #     tmp_flds = hdr1.split(' ')
    #     tmp_flds.remove('Date')
    #
    #     flds = {nn+3: hh for nn, hh in enumerate(tmp_flds)}
    #
    #     # poi_flds maps column index to POI and is used to rename the dataframe columns from indices to station IDs
    #     poi_flds = dict()
    #
    #     # poi_seg_flds maps POI to the related segment ID
    #     poi_seg_flds = dict()
    #
    #     for xx, yy in flds.items():
    #         tfld = yy.split('_')
    #         segid = int(tfld[2]) - 1  # Change to zero-based indices
    #         poiid = tfld[4]
    #
    #         poi_flds[xx] = poiid
    #         poi_seg_flds[poiid] = segid
    #
    #     return poi_flds, poi_seg_flds
    #
    # @staticmethod
    # def _read_streamflow_ascii(filename, field_names):
    #     """Read the simulated streamflow from a PRMS CSV model output file"""
    #     df = pd.read_csv(filename, sep=r'\s+', header=None, skiprows=2, parse_dates={'time': [0, 1, 2]},
    #                      index_col='time')
    #
    #     df.rename(columns=field_names, inplace=True)
    #
    #     return df

    def set_gis(self, filename: str,
                hru_layer: Optional[str] = None,
                hru_key: Optional[str] = None,
                seg_layer: Optional[str] = None,
                seg_key: Optional[str] = None,
                ):

        if hru_layer:
            if self.__verbose:
                print('Reading HRU polygons')
            self.__hru_poly = read_gis(filename, hru_layer)
            self.__hru_shape_key = hru_key

        if seg_layer:
            if self.__verbose:
                print('Reading segment lines')
            self.__seg_poly = read_gis(filename, seg_layer)
            self.__seg_shape_key = seg_key

    def plot(self, name: str,
             output_dir: Optional[str] = None,
             limits: Optional[Union[str, List[float], Tuple[float, float]]] = 'valid',
             mask_defaults: Optional[str] = None,
             **kwargs):
        """Plot an output variable.
        """

        var_data = self.get(name).iloc[0, :].to_frame(name=name)

        if isinstance(limits, str):
            # if limits == 'valid':
            #     # Use the defined valid range of possible values
            #     drange = [cparam.minimum, cparam.maximum]
            # elif limits == 'centered':
            #     # Use the maximum range of the actual data values
            #     lim = max(abs(cparam.data.min().min()), abs(cparam.data.max().max()))
            #     drange = [-lim, lim]
            if limits == 'absolute':
                # Use the min and max of the data values
                drange = [var_data.min().min(), var_data.max().max()]
            else:
                raise ValueError('String argument for limits must be "valid", "centered", or "absolute"')
        elif isinstance(limits, (list, tuple)):
            if len(limits) != 2:
                raise ValueError('When a list is used for plotting limits it should have 2 values (min, max)')

            drange = [min(limits), max(limits)]
        else:
            raise TypeError('Argument, limits, must be string or a list[min,max]')

        cmap, norm = set_colormap(name, var_data, min_val=drange[0],
                                  max_val=drange[1], **kwargs)

        if self.__hru_poly is not None:
            # Get extent information
            minx, miny, maxx, maxy = self.__hru_poly.geometry.total_bounds

            crs_proj = get_projection(self.__hru_poly)

            # Takes care of multipolygons that are in the NHM geodatabase/shapefile
            geoms_exploded = self.__hru_poly.explode(index_parts=True).reset_index(level=1, drop=True)

            # print('Writing first plot')
            df_mrg = geoms_exploded.merge(var_data, left_on=self.__hru_shape_key, right_index=True, how='left')

            fig_width, fig_height = get_figsize([minx, maxx, miny, maxy], **dict(kwargs))
            kwargs.pop('init_size', None)

            fig = plt.figure(figsize=(fig_width, fig_height))

            ax = plt.axes(projection=crs_proj)

            # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(30, 20))

            ax = plt.axes(projection=crs_proj)

            try:
                ax.coastlines()
            except AttributeError:
                pass

            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
            gl.top_labels = None
            gl.right_labels = None
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            # ax.gridlines()
            ax.set_extent([minx, maxx, miny, maxy], crs=crs_proj)

            mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            mapper.set_array(df_mrg[name])
            cax = fig.add_axes((ax.get_position().x1 + 0.01,
                                ax.get_position().y0, 0.02,
                                ax.get_position().height))

            plt.colorbar(mapper, cax=cax)   # , label=cparam.units)
            plt.title(f'Variable: {name}')

            col = plot_polygon_collection(ax, df_mrg.geometry, values=df_mrg[name],
                                          **dict(kwargs, cmap=cmap, norm=norm))
