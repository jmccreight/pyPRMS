#!/usr/bin/env python3

import numpy as np
from typing import Dict, List, Optional, Sequence, Union

from pyPRMS.constants import DATA_TYPES


class ControlVariable(object):
    """
    Class object for a single control variable.
    """

    # Author: Parker Norton (pnorton@usgs.gov)
    # Create date: 2019-04-18

    def __init__(self, name: Optional[str] = None,
                 datatype: Optional[int] = None,
                 default: Optional[Union[int, float, str]] = None,
                 description: Optional[str] = None,
                 valid_values: Optional[Union[int, float, str]] = None,
                 value_repr: Optional[str] = None):
        """Initialize a control variable object.

        :param name: Name of control variable
        :param datatype: The datatype of the control variable
        :param default: The default value
        :param description: The description of the control variable
        :param valid_values: The valid values
        :param value_repr: What do the valid values represent (e.g. flag, parameter, etc)
        """

        self.__name = name
        self.__datatype = datatype
        self.__default = default
        self.__description = description
        self.__force_default = False
        self.__valid_values = valid_values  # Possible valid values
        self.__value_repr = value_repr  # What do the valid_values represent (e.g. flag, parameter, etc)?
        self.__associated_value = None  # Based on a value what's the associated valid_value?
        self.__values = None

    def __str__(self) -> str:
        outstr = f'name: {self.name}\ndatatype: {self.datatype}\n'

        if self.default is not None:
            outstr += f'default: {self.default}\n'

        outstr += 'Size of data: '
        if self.values is not None:
            outstr += f'{self.size}\n'
        else:
            outstr += '<empty>\n'

        return outstr

    @property
    def associated_values(self) -> List[str]:
        """Get list of control variable names which are associated with this
        control variable.

        :returns: Associated control variables
        """

        assoc_vals = []
        if self.size > 1:
            for xx in self.values:
                for vv in self.__valid_values[xx]:
                    assoc_vals.append(vv)
        else:
            for vv in self.__valid_values[str(self.values)]:
                assoc_vals.append(vv)

        return assoc_vals

    @property
    def datatype(self) -> int:
        """Get the datatype of the control variable.

        :returns: datatype
        """

        return self.__datatype

    @datatype.setter
    def datatype(self, dtype: int):
        """Sets the datatype of the control variable.

        :param dtype: The datatype for the control variable (1-Integer, 2-Float, 3-Double, 4-String)
        """

        if dtype in DATA_TYPES:
            self.__datatype = dtype
        else:
            print(f'WARNING: Datatype, {dtype}, is not valid.')

    @property
    def default(self) -> Union[int, float, str, None]:
        """Get default value for control variable.

        :returns: current default value
        """

        if self.__default is not None:
            if self.__default.size > 1:
                return self.__default
            else:
                return self.__default[0]
        else:
            return None

    @default.setter
    def default(self, value: Union[int, float, str]):
        """Set the default value for the control variable.

        :param value: The default value
        """

        # Convert datatype first
        datatype_conv = {1: self.__str_to_int, 2: self.__str_to_float,
                         3: self.__str_to_float, 4: self.__str_to_str}

        if self.__datatype in DATA_TYPES.keys():
            value = datatype_conv[self.__datatype](value)
        else:
            err_txt = f'Defined datatype {self.__datatype} for control variable {self.__name} is not valid'
            raise TypeError(err_txt)

        self.__default = np.array(value)

    @property
    def force_default(self) -> bool:
        """Get logical value which indicates whether the default value for a
        control variable should always be used instead of the current value.

        """

        return self.__force_default

    @force_default.setter
    def force_default(self, value: Union[bool, int]):
        """Set (or unset) forced use of default value.

        :param value: new force_default value
        """

        self.__force_default = bool(value)

    @property
    def name(self) -> str:
        """Returns the name of the control variable.

        :returns: Name of control variable
        """

        return self.__name

    @property
    def size(self) -> int:
        """Get the number of values for the control variable.

        :returns: Number of values
        """

        if self.__values is not None:
            return self.__values.size
        elif self.__default is not None:
            return self.__default.size
        else:
            return 0

    @property
    def valid_values(self) -> Dict:
        """Get the values that are valid for the control variable.

        :returns: Valid values for the control variable
        """

        return self.__valid_values

    @valid_values.setter
    def valid_values(self, data: Dict):
        """Set the values that are valid for the control variable.

        :param data: Valid values for the control variable
        """

        if isinstance(data, dict):
            self.__valid_values = data

    @property
    def value_repr(self) -> str:
        """Get what the control variable value represents.

        A control variable value can represent a flag, interval, or parameter.

        :returns: Control variable representation value
        """

        return self.__value_repr

    @value_repr.setter
    def value_repr(self, data: Union[str, None]):
        """Set the control variable representation.

        :param data: Representation value
        """

        self.__value_repr = data

    @property
    def values(self) -> Union[List[str], List[int], int, float, str]:
        """Get the values for the control variable.

        If force_default is True then the default value is returned regardless
        of what the value is set to; otherwise, current value is returned.

        :returns: Value(s) of control variable
        """

        if self.__values is not None:
            if self.__force_default:
                return self.default
            elif self.__values.size > 1:
                return self.__values
            else:
                return self.__values[0]
        else:
            return self.default

    @values.setter
    def values(self, data: Union[Sequence[str], str]):
        """Set the values for the control variable.

        :param data: New value(s)
        """

        # Convert datatype first
        datatype_conv = {1: self.__str_to_int, 2: self.__str_to_float,
                         3: self.__str_to_float, 4: self.__str_to_str}

        if self.__datatype in DATA_TYPES.keys():
            data = datatype_conv[self.__datatype](data)
        else:
            raise TypeError(f'Defined datatype {self.__datatype} for parameter {self.__name} is not valid')

        # Convert to ndarray
        self.__values = np.array(data)

    @staticmethod
    def __str_to_float(data: Union[List[str], str]) -> List[float]:
        """Convert strings to a floats.

        :param data: value(s)

        :returns: Array of floats
        """

        # Convert provide list of data to float
        if isinstance(data, str):
            return [float(data)]
        else:
            try:
                return [float(vv) for vv in data]
            except ValueError as ve:
                print(ve)

    @staticmethod
    def __str_to_int(data: Union[List[str], str]) -> List[int]:
        """Converts strings to integers.

        :param data: value(s)

        :returns: array of integers
        """

        if isinstance(data, str):
            return [int(data)]
        else:
            # Convert list of data to integer
            try:
                return [int(vv) for vv in data]
            except ValueError as ve:
                print(ve)

    @staticmethod
    def __str_to_str(data: Union[List[str], str]) -> List[str]:
        """Null op for string-to-string conversion.

        :param data: value(s)

        :returns: unmodified array of data
        """

        # nop for list of strings
        if isinstance(data, str):
            data = [data]

        # 2019-05-22 PAN: For python 3 force string type to byte
        #                 otherwise they are treated as unicode
        return data
        # return [dd.encode() for dd in data]
