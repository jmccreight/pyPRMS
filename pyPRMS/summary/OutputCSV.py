import pandas as pd


class OutputCSV(object):
    """Class for working with PRMS CSV output files.
    """

    def __init__(self, filename: str, on_off: int):
        """Initialize the OutputCSV object.

        :param filename: Name of the PRMS CSV output file
        :param on_off: ON_OFF flag for the PRMS model (1=comma-sep (basin vars + streamflow), 2=space-sep (streamflow only))
        """
        self.__filename = filename
        self.__on_off = on_off
        self.__data = None

    def _read_streamflow_header(self, filename):
        """Read the headers from a PRMS CSV model output file"""

        fhdl = open(filename, 'r')

        # First and second rows are headers
        hdr1 = fhdl.readline().strip()

        fhdl.close()

        if self.__on_off == 1:
            # This is a comma-separated file that includes a selection
            # of basin variables and simulated streamflow at each POI.
            pass
        elif self.__on_off == 2:
            # This is a space-separated file that includes only the simulated
            # streamflow at each POI.
            tmp_flds = hdr1.split(' ')
            tmp_flds.remove('Date')

            flds = {nn+3: hh for nn, hh in enumerate(tmp_flds)}

            # poi_flds maps column index to POI and is used to rename the dataframe columns from indices to station IDs
            poi_flds = dict()

            # poi_seg_flds maps POI to the related segment ID
            poi_seg_flds = dict()

            for xx, yy in flds.items():
                tfld = yy.split('_')
                segid = int(tfld[2]) - 1  # Change to zero-based indices
                poiid = tfld[4]

                poi_flds[xx] = poiid
                poi_seg_flds[poiid] = segid

        return poi_flds, poi_seg_flds

    @staticmethod
    def _read_streamflow_ascii(filename, field_names):
        """Read the simulated streamflow from a PRMS CSV model output file"""
        df = pd.read_csv(filename, sep=r'\s+', header=None, skiprows=2, parse_dates={'time': [0, 1, 2]},
                         index_col='time')

        df.rename(columns=field_names, inplace=True)

        return df