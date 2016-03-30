#!/usr/bin/env python

# Process AET datasets to create a range of values to use for PRMS calibration by HRU. 
# Individual datasets by HRU are created.

from __future__ import print_function
# from builtins import range
# from six import iteritems

import pandas as pd
import datetime
import sys

# Regions in the National Hydrology Model
regions = ('01', '02', '03', '04', '05', '06', '07', '08',
           '09', '10L', '10U', '11', '12', '13', '14', '15',
           '16', '17', '18')

# Number of HRUs in each region
hrus_by_region = (2462, 4827, 9899, 5936, 7182, 2303, 8205, 4449,
                  1717, 8603, 10299, 7373, 7815, 1958, 3879, 3441,
                  2664, 11102, 5837)


# NOTE: These AET files were generated by Paul Micheletty and use units of mm

def load_mod16_aet(filename, st_date, en_date, missing_val=[-9999.0]):
    # ---------------------------------------------------------------------------
    # Process MOD16 AET
    #
    # Parameters:
    # filename      Full path of file to read
    # missing_val   One or more values used for a missing value
    # st_date       Start date for returned dataset.
    #               Can be datetime or a string of form 'YYYY-MM-DD'
    # en_date       End date for returned dataset.
    #               Can be datetime or a string of form 'YYYY-MM-DD'

    #file1 = 'MOD16.CONUS'
    #missing = [-9999.0]

    if not isinstance(missing_val, tuple):
        missing_val = list(missing_val)

    if not isinstance(st_date, datetime.datetime):
        date_split = st_date.split('-')
        st_date = datetime.datetime(date_split[0], date_split[1], date_split[2])

    if not isinstance(en_date, datetime.datetime):
        date_split = en_date.split('-')
        en_date = datetime.datetime(date_split[0], date_split[1], date_split[2])

    ds1 = pd.read_csv(filename, sep=' ', 
                      skipinitialspace=True, na_values=missing_val, engine='c')

    # Transpose the dataframe: dates on rows, hrus on columns
    ds1b = ds1.unstack().unstack()

    ds1b.reset_index(inplace=True)
    ds1b['thedate'] = pd.to_datetime(ds1b.ix[:, 0], format='%Y.%m')
    ds1b.set_index(['thedate'], inplace=True)
    ds1b.drop(['index'], axis=1, inplace=True)

    # Monthly values are aligned at the start of the month, change to the end of the month
    ds1c = ds1b.resample('M', how='mean')

    # Make the HRU column labels one-based
    ds1c.rename(columns=lambda x: ds1c.columns.get_loc(x)+1, inplace=True)
    ds1c.head()

    return ds1c[st_date:en_date]


def load_ssebop_aet(filename, st_date, en_date, missing_val=[-9999.0]):
    # ---------------------------------------------------------------------------
    # Process SSEBop AET
    #
    # Parameters:
    # filename      Name of file to read
    # missing_val   One or more values used for a missing value
    # st_date       Start date for returned dataset.
    #               Can be datetime or a string of form 'YYYY-MM-DD'
    # en_date       End date for returned dataset.
    #               Can be datetime or a string of form 'YYYY-MM-DD'

    #file3 = 'SSEBop.CONUS'
    #missing = [-9999.0]

    if not isinstance(missing_val, tuple):
        missing_val = list(missing_val)

    if not isinstance(st_date, datetime.datetime):
        date_split = st_date.split('-')
        st_date = datetime.datetime(date_split[0], date_split[1], date_split[2])

    if not isinstance(en_date, datetime.datetime):
        date_split = en_date.split('-')
        en_date = datetime.datetime(date_split[0], date_split[1], date_split[2])

    # Read the dataset
    ds3 = pd.read_csv(filename, sep=' ', 
                      skipinitialspace=True, na_values=missing_val, engine='c')

    # Transpose the dataframe: dates on rows, hrus on columns
    ds3b = ds3.unstack().unstack()

    ds3b.reset_index(inplace=True)
    ds3b['thedate'] = pd.to_datetime(ds3b.ix[:,0], format='%Y.%m')
    ds3b.set_index(['thedate'], inplace=True)
    ds3b.drop(['index'], axis=1, inplace=True)

    # Monthly values are aligned at the start of the month, change to the end of the month
    ds3c = ds3b.resample('M', how='mean')

    # Make the HRU column labels one-based
    ds3c.rename(columns=lambda x: ds3c.columns.get_loc(x)+1, inplace=True)
    ds3c.head()

    return ds3c[st_date:en_date]


def load_mwbm_aet(filename, st_date, en_date, missing_val=[-9999.0]):
    # ---------------------------------------------------------------------------
    # Process MWBM AET
    #
    # Parameters:
    # filename      Name of file to read
    # missing_val   One or more values used for a missing value
    # st_date       Start date for returned dataset.
    #               Can be datetime or a string of form 'YYYY-MM-DD'
    # en_date       End date for returned dataset.
    #               Can be datetime or a string of form 'YYYY-MM-DD'

    #file2 = 'MWBM.CONUS'
    #missing = [-9999.0]

    if not isinstance(missing_val, tuple):
        missing_val = list(missing_val)

    if not isinstance(st_date, datetime.datetime):
        date_split = st_date.split('-')
        st_date = datetime.datetime(date_split[0], date_split[1], date_split[2])

    if not isinstance(en_date, datetime.datetime):
        date_split = en_date.split('-')
        en_date = datetime.datetime(date_split[0], date_split[1], date_split[2])

    # ---------------------------------------------------------------------------
    # Process MWBM AET files
    ds2 = pd.read_csv(filename, sep=' ', skipinitialspace=True, 
                      na_values=missing_val, engine='c')

    ds2b = ds2.unstack().unstack()
    ds2c = ds2b.reset_index()

    # The MWBM data doesn't provide the datetime information for some reason
    # Create a range from 1949-1 to 2010-12, add it to the dataframe, 
    # and make it the index
    rng = pd.date_range('1/1/1949', periods=744, freq='M')

    ds2c['thedate'] = rng
    ds2c.set_index(['thedate'], inplace=True)
    ds2c.drop(['index'], axis=1, inplace=True)

    # Make the HRU column labels one-based
    ds2c.rename(columns=lambda x: ds2c.columns.get_loc(x)+1, inplace=True)
    ds2c.head()

    return ds2c[st_date:en_date]


def pull_by_hru(src_dir, dst_dir, st_date, en_date, region):
    # For a given region pull AET for each HRU and write it to the dst_dir
    #
    # Parameters:
    # srcdir    Location of the AET datasets
    # dstdir    Top-level location to write HRUs
    # st_date   Start date for output dataset
    # en_date   End date for output datasdet
    # region    The region to pull HRUs out of

    if not isinstance(st_date, datetime.datetime):
        date_split = st_date.split('-')
        st_date = datetime.datetime(date_split[0], date_split[1], date_split[2])

    if not isinstance(en_date, datetime.datetime):
        date_split = en_date.split('-')
        en_date = datetime.datetime(date_split[0], date_split[1], date_split[2])

    # Get the zero-based start and end index for the selected region
    start_idx = sum(hrus_by_region[0:regions.index(region)])
    end_idx = (sum(hrus_by_region[0:regions.index(region)]) + hrus_by_region[regions.index(region)] - 1)

    # Load each dataset into memory. These datasets contain all regions
    print("Loading:")
    # Parser for the date information
    parser = lambda x: pd.to_datetime(x, format='%Y-%m-%d')

    # When using the CONUS file we have to pull the region from it
    region_start = sum(hrus_by_region[0:regions.index(region)]) + 1
    region_end = region_start + hrus_by_region[regions.index(region)]

    # Build the set of columns to load
    # Column 0 is the date and is always included
    column_set = [0]
    column_set.extend(range(region_start, region_end))

    print("\tMOD16 AET..")
    aet_mod16 = pd.read_csv('%s/MOD16_AET_CONUS_2000-2010' % src_dir, sep=' ', parse_dates=True, date_parser=parser,
                            index_col='thedate', usecols=column_set)

    print("\tSSEBop AET..")
    aet_SSEBop = pd.read_csv('%s/SSEBop_AET_CONUS_2000-2010' % src_dir, sep=' ', parse_dates=True, date_parser=parser,
                             index_col='thedate', usecols=column_set)

    print("\tMWBM AET..")
    aet_mwbm = pd.read_csv('%s/MWBM_AET_CONUS_2000-2010' % src_dir, sep=' ', parse_dates=True, date_parser=parser,
                           index_col='thedate', usecols=column_set)


    # aet_mod16 = load_mod16_aet('%s/MOD16_AET_CONUS_2000-2010' % src_dir, st_date, en_date)
    # aet_mod16 *= 0.0393700787   # convert to inches
    #
    # print("\tSSEBop AET..")
    # aet_SSEBop = load_ssebop_aet('%s/SSEBop_AET_CONUS_2000-2010' % src_dir, st_date, en_date)
    # aet_SSEBop *= 0.0393700787   # convert to inches
    #
    # print("\tMWBM AET..")
    # aet_mwbm = load_mwbm_aet('%s/MWBM_AET_CONUS_2000-2010   ' % src_dir, st_date, en_date)
    # aet_mwbm *= 0.0393700787   # convert to inches

    # hru_index is not zero-based, it is one-based; it is referencing the column names not the position
    #hru_index = start_idx + 1

    print("Writing out HRUs:")
    for hh in range(start_idx+1, end_idx+2):
        sys.stdout.write('\r\t%06d ' % (hh-start_idx-1))
        sys.stdout.flush()

        # ---------------------------------------------------------------------------
        # Retrieve single HRU from each input model
        # ---------------------------------------------------------------------------
        modis_ss = pd.DataFrame(aet_mod16.ix[:,hh])
        modis_ss.rename(columns={modis_ss.columns[0]: 'modis'}, inplace=True)

        ssebop_ss = pd.DataFrame(aet_SSEBop.ix[:,hh])
        ssebop_ss.rename(columns={ssebop_ss.columns[0]: 'ssebop'}, inplace=True)

        mwbm_ss = pd.DataFrame(aet_mwbm.ix[:,hh])
        mwbm_ss.rename(columns={mwbm_ss.columns[0]: 'mwbm'}, inplace=True)

        # ---------------------------------------------------------------------------
        # Join modis, ssebop, and mwbm together
        # ---------------------------------------------------------------------------
        ds_join = modis_ss.join(ssebop_ss, how='outer')
        ds_join = ds_join.join(mwbm_ss, how='outer')
        #print ds_join.head()

        # Create min and max fields based on the dataset values
        # Modify dateframe to the format required by PRMS for calibration
        ds_join['min'] = ds_join.min(axis=1)
        ds_join['max'] = ds_join.max(axis=1)
        ds_join.drop(['modis', 'ssebop', 'mwbm'], axis=1, inplace=True)
        ds_join['year'] = ds_join.index.year
        ds_join['month'] = ds_join.index.month
        ds_join.reset_index(inplace=True)
        ds_join.drop(['thedate'], axis=1, inplace=True)

        # Output HRU into the correct HRU directory
        # outfile format: <dst_dir>/r10U_000000/AETerror
        # HRU numbers are relative to the selected region
        outfile = '%s/r%s_%06d/AETerror' % (dst_dir, region, (hh-start_idx-1))
        ds_join.to_csv(outfile, sep=' ', float_format='%0.5f', columns=['year', 'month', 'min', 'max'], 
                       header=False, index=False)

    print('')


def pull_by_hru_GCPO(src_dir, dst_dir, st_date, en_date, region):
    # For a given region pull AET for each HRU and write it to the dst_dir
    #
    # Parameters:
    # srcdir    Location of the AET datasets
    # dstdir    Top-level location to write HRUs
    # st_date   Start date for output dataset
    # en_date   End date for output datasdet
    # region    The region to pull HRUs out of

    # Override the region information for the GCPO
    regions = ['GCPO']
    hrus_by_region = [20251]

    if not isinstance(st_date, datetime.datetime):
        date_split = st_date.split('-')
        st_date = datetime.datetime(date_split[0], date_split[1], date_split[2])

    if not isinstance(en_date, datetime.datetime):
        date_split = en_date.split('-')
        en_date = datetime.datetime(date_split[0], date_split[1], date_split[2])

    # Get the zero-based start and end index for the selected region
    start_idx = sum(hrus_by_region[0:regions.index(region)])
    # end_idx = (sum(hrus_by_region[0:regions.index(region)]) + hrus_by_region[regions.index(region)] - 1)

    # Load each dataset into memory. These datasets contain HRUs for the GCPO study area
    print("Loading:")

    # Parser for the date information
    parser = lambda x: pd.to_datetime(x, format='%Y-%m-%d')

    print("\tMOD16 AET..")
    aet_mod16 = pd.read_csv('%s/MOD16_AET_GCPO_2000-2010' % src_dir, sep=' ', parse_dates=True, date_parser=parser,
                            index_col='thedate')

    print("\tSSEBop AET..")
    aet_SSEBop = pd.read_csv('%s/SSEBop_AET_GCPO_2000-2010' % src_dir, sep=' ', parse_dates=True, date_parser=parser,
                             index_col='thedate')

    print("\tMWBM AET..")
    aet_mwbm = pd.read_csv('%s/MWBM_AET_GCPO_2000-2010' % src_dir, sep=' ', parse_dates=True, date_parser=parser,
                           index_col='thedate')

    print("Writing out HRUs:")
    for hh in range(hrus_by_region[regions.index(region)]):
    # for hh in xrange(start_idx+1, end_idx+2):
    #     sys.stdout.write('\r\t%06d ' % (hh-start_idx-1))
        sys.stdout.write('\r\t%06d ' % hh)
        sys.stdout.flush()

        # ---------------------------------------------------------------------------
        # Retrieve single HRU from each input model
        # ---------------------------------------------------------------------------
        modis_ss = pd.DataFrame(aet_mod16.iloc[:,hh])
        modis_ss.rename(columns={modis_ss.columns[0]: 'modis'}, inplace=True)

        ssebop_ss = pd.DataFrame(aet_SSEBop.iloc[:,hh])
        ssebop_ss.rename(columns={ssebop_ss.columns[0]: 'ssebop'}, inplace=True)

        mwbm_ss = pd.DataFrame(aet_mwbm.iloc[:,hh])
        mwbm_ss.rename(columns={mwbm_ss.columns[0]: 'mwbm'}, inplace=True)

        # ---------------------------------------------------------------------------
        # Join modis, ssebop, and mwbm together
        # ---------------------------------------------------------------------------
        ds_join = modis_ss.join(ssebop_ss, how='outer')
        ds_join = ds_join.join(mwbm_ss, how='outer')

        # Create min and max fields based on the dataset values
        # Modify dataframe to the format required by PRMS for calibration
        ds_join['min'] = ds_join.min(axis=1)
        ds_join['max'] = ds_join.max(axis=1)
        ds_join.drop(['modis', 'ssebop', 'mwbm'], axis=1, inplace=True)
        ds_join['year'] = ds_join.index.year
        ds_join['month'] = ds_join.index.month
        ds_join.reset_index(inplace=True)
        ds_join.drop(['thedate'], axis=1, inplace=True)

        # Output HRU into the correct HRU directory
        # outfile format: <dst_dir>/r10U_000000/AETerror
        # HRU numbers are relative to the selected region
        outfile = '%s/r%s_%06d/AETerror' % (dst_dir, region, (hh-start_idx))
        ds_join.to_csv(outfile, sep=' ', float_format='%0.5f', columns=['year', 'month', 'min', 'max'],
                       header=False, index=False)

    print('')


# ===========================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description='Split MWBM model output into individual HRUs')
    parser.add_argument('-b', '--basedir', help='Base directory for regions', required=True)
    parser.add_argument('-s', '--srcdir', help='Source data directory', required=True)
    parser.add_argument('-r', '--region', help='Region to process', required=True)
    parser.add_argument('--range', help='Create error range files', action='store_true')

    args = parser.parse_args()

    selected_region = args.region
    base_dir = args.basedir
    src_dir = args.srcdir
    dst_dir = '%s/r%s_byHRU' % (base_dir, args.region)

    # The HRUs for a particular region can be processed
    # selected_region = 'GCPO'
    # src_dir = '/media/scratch/PRMS/datasets/AET'
    # dst_dir = '/media/scratch/PRMS/regions/r%s_byHRU' % selected_region

    st = datetime.datetime(2000, 1, 1)
    en = datetime.datetime(2010, 12, 31)

    # start_idx = sum(hrus_by_region[0:regions.index(selected_region)])
    # end_idx = (sum(hrus_by_region[0:regions.index(selected_region)]) + \
    #                hrus_by_region[regions.index(selected_region)] - 1)
    #
    # print 'Total number of regions:', len(regions)
    # print 'Selected region:', selected_region
    # print 'Total HRUs:', sum(hrus_by_region)
    # print 'Starting zero-based index in NHM for region %s:' % selected_region, start_idx
    # print 'Ending zero-based index in NHM for region %s:' % selected_region, end_idx

    if selected_region == 'GCPO':
        pull_by_hru_GCPO(src_dir, dst_dir, st, en, selected_region)
    else:
        pull_by_hru(src_dir, dst_dir, st, en, selected_region)


if __name__ == '__main__':
    main()
