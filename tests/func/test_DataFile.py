# import pytest
# import numpy as np

from pyPRMS import DataFile


class TestStreamflow:

    def test_read_datafile_single_station(self, datadir):
        sf_filename = datadir / 'sf_data_pipestem_bandit'

        datafile = DataFile(sf_filename, verbose=False)
        obs_sf = datafile.get('runoff')

        expected_stations = ['06469400']

        assert obs_sf.data.mean().values[0] == 30.094350205198356
        assert len(obs_sf.data.columns) == 1
        assert len(obs_sf.data) == 731
        assert obs_sf.units == 'cfs'
        assert list(obs_sf.data.columns) == expected_stations

    def test_read_datafile_multiple_stations(self, datadir):
        sf_filename = datadir / 'sf_data_downsizer'

        datafile = DataFile(sf_filename, verbose=False)
        obs_sf = datafile.get('runoff')

        expected_mean = {'14142500': 0.0,
                         '14137002': 0.0,
                         '14137000': 2667.07972451021,
                         '14134000': 178.83350439514788,
                         '14141500': 341.5708917526521,
                         '14140000': 1432.5777467542205,
                         '14139800': 283.73439571085805,
                         '14139700': 242.826600224096,
                         '14138850': 889.0568331357849,
                         '14138800': 222.1890569061978,
                         '14138720': 0.0,
                         '14138900': 259.6904311639005,
                         '14138870': 160.55626150658415,
                         '14142800': 0.0}

        expected_stations = ['14142500',
                             '14137002',
                             '14137000',
                             '14134000',
                             '14141500',
                             '14140000',
                             '14139800',
                             '14139700',
                             '14138850',
                             '14138800',
                             '14138720',
                             '14138900',
                             '14138870',
                             '14142800']

        assert obs_sf.data.describe().mean().to_dict() == expected_mean
        assert len(obs_sf.data.columns) == 14   # number of stations
        assert len(obs_sf.data) == 731   # number of days
        assert obs_sf.units == 'cfs'
        assert list(obs_sf.data.columns) == expected_stations

    def test_read_datafile_sagehen(self, datadir):
        sf_filename = datadir / 'sagehen.data'

        datafile = DataFile(sf_filename, verbose=False)
        obs_sf = datafile.get('runoff')

        # assert obs_sf.data.describe().mean().to_dict() == expected_mean
        assert len(datafile.data.columns) == 7   # number of stations
        assert len(datafile.data) == 8608   # number of days
        assert list(datafile.input_variables.keys()) == ['tmax', 'tmin', 'precip', 'runoff']
        assert obs_sf.units is None
        # assert obs_sf.get('runoff').get('stations') is None
