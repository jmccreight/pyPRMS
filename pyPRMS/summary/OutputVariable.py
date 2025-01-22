import pandas as pd


class OutputVariable(object):
    """Container for a single output variable
    """

    def __init__(self, name: str, filename: str, metadata: dict):
        """Initialize the OutputVariable object.

        :param name: Name of the output variable
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
    def filename(self):
        """Return the path to the model output variable
        """
        return self.__filename

    def _read_file(self):
        """Read model output file
        """

        self.__data = pd.read_csv(self.__filename, sep=',', skipinitialspace=True,
                                  header=0, index_col=0, parse_dates=True)

        self.__data.index.name = 'time'

        if self.metadata['dimensions'][0] in ['nhru', 'nsegment', 'nsub']:
            self.__data.columns = self.__data.columns.astype(int)
        elif self.metadata['dimensions'][0] == 'one':
            self.__data = self.__data.loc[:, self.__name].to_frame()
