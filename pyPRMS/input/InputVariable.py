import pandas as pd

from typing import Optional


class InputVariable(object):
    """Class for working with input variables."""

    def __init__(self, name: str,
                 data: pd.DataFrame,
                 units: Optional[str] = None):
        """Initialize the InputVariable object.

        :param name: Name or kind of the input variable
        :param data: Input variable data
        :param units: Units of the input variable
        """
        self.__name = name
        self.__units = units
        self.data = data

    @property
    def data(self) -> pd.DataFrame:
        """Returns the input variable data.

        :returns: Input variable dataframe
        """

        return self.__data

    @data.setter
    def data(self, data_in: pd.DataFrame):
        """Set the input variable data.

        :param data_in: Input variable data
        """

        col_names = {}
        for xx in data_in.columns:
            col_names[xx] = xx.split('_')[1]

        self.__data = data_in.copy()
        self.__data.rename(columns=col_names, inplace=True)

    @property
    def name(self) -> str:
        """Returns the input variable kind.

        :returns: Input variable kind
        """

        return self.__name

    @property
    def units(self) -> str:
        """Returns the input variable units.

        :returns: Input variable units
        """

        return self.__units
