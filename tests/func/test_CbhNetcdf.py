import pytest
import os
from distutils import dir_util
from pyPRMS import CbhNetcdf

@pytest.fixture
def datadir(tmpdir, request):
    """
    Fixture responsible for searching a folder with the same name of test
    module and, if available, moving all contents to a temporary directory so
    tests can use them freely.
    """
    # 2023-07-18
    # https://stackoverflow.com/questions/29627341/pytest-where-to-store-expected-data
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))

    return tmpdir


class TestCbhNetcdf:

    def test_read_netcdf_roundtrip_ascii(self, datadir, tmp_path):
        """Test reading a netCDF file and writing it to an ASCII file"""
        cbh_file = str(datadir.join('cbh.nc'))

        nhm_ids = [57863, 57864, 57867, 57868, 57869, 57872, 57873, 57874, 57877, 57878, 57879, 57880, 57881, 57882]
        cbh = CbhNetcdf(cbh_file, nhm_hrus=nhm_ids)

        out_path = tmp_path / 'run_files'
        out_path.mkdir()

        for cvar in ['prcp', 'tmax', 'tmin']:
            out_file = out_path / f'{cvar}_chk.day'
            cbh.write_ascii(out_file, variable=cvar)

            with open(datadir.join(f'{cvar}.day'), 'r') as f:
                lines_orig = f.readlines()

            with open(out_file, 'r') as f:
                lines_chk = f.readlines()

            assert lines_orig == lines_chk

    def test_read_netcdf_roundtrip_netcdf(self, datadir, tmp_path):
        """Test reading a netCDF file and writing it to a netCDF file"""
        cbh_file = str(datadir.join('cbh.nc'))

        nhm_ids = [57863, 57864, 57867, 57868, 57869, 57872, 57873, 57874, 57877, 57878, 57879, 57880, 57881, 57882]
        cbh = CbhNetcdf(cbh_file, nhm_hrus=nhm_ids)

        out_path = tmp_path / 'run_files'
        out_path.mkdir()
        out_file = out_path / 'cbh_chk.nc'

        cbh.write_netcdf(out_file, global_attrs=dict(source='test'))
        cbh_chk = CbhNetcdf(out_file, nhm_hrus=nhm_ids)

        assert cbh_chk.data.equals(cbh.data)
        assert cbh_chk.data.attrs['source'] == 'test'
