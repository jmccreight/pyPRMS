import pytest
import datetime
# import decimal
import numpy as np

from pyPRMS.prms_helpers import flex_type, set_date


class TestPrmsHelpers:

    def test_flex_type_invalid(self):
        with pytest.raises(ValueError):
            flex_type(1e9999999999999999999)

    def test_flex_type_int(self):
        result = flex_type(1)
        assert result == '1'

    def test_flex_type_float(self):
        result = flex_type(1.0)
        assert result == '1.0'

        result = flex_type(0.000001)
        assert result == '0.000001'

        result = flex_type(1e-6)
        assert result == '0.000001'

    def test_flex_type_str(self):
        result = flex_type('1')
        assert result == '1'

    def test_set_date_w_date(self):
        """Test the set_date function."""

        # Test with a datetime.date object
        date = datetime.date(2020, 1, 1)

        expected = datetime.datetime(2020, 1, 1, 0, 0)
        result = set_date(date)
        assert result == expected

    def test_set_date_w_datetime(self):
        """Test the set_date function."""

        # Test with a datetime.date object
        date = datetime.datetime(2020, 1, 1)

        expected = datetime.datetime(2020, 1, 1, 0, 0)
        result = set_date(date)
        assert result == expected

    def test_set_date_w_str(self):
        """Test the set_date function."""

        expected = datetime.datetime(2020, 1, 1, 0, 0)

        result = set_date('2020-1-1')
        assert result == expected

        result = set_date('2020-01-01')
        assert result == expected

        result = set_date('2020-01-01 00:00:00')
        assert result == expected

    def test_set_date_w_ndarray(self):
        """Test the set_date function."""

        expected = datetime.datetime(2020, 1, 1, 0, 0)

        result = set_date(np.array([2020, 1, 1]))
        assert result == expected

        result = set_date(np.array([2020, 1, 1, 0, 0, 0]))
        assert result == expected

        # # Test with a list of dates
        # date = [datetime(2020, 1, 1), datetime(2020, 1, 2)]
        # expected = np.array([20200101, 20200102])
        # result = set_date(date)
        # np.testing.assert_array_equal(result, expected)
        #
        # # Test with a numpy array of dates
        # date = np.array([datetime(2020, 1, 1), datetime(2020, 1, 2)])
        # expected = np.array([20200101, 20200102])
        # result = set_date(date)
        # np.testing.assert_array_equal(result, expected)
