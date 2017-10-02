
from __future__ import (absolute_import, division, print_function)
from future.utils import iteritems

import xml.dom.minidom as minidom
import xml.etree.ElementTree as xmlET

from pyPRMS.Parameters import Parameters
from pyPRMS.Dimensions import Dimensions
from pyPRMS.constants import NHM_DATATYPES, PARAMETERS_XML, DIMENSIONS_XML


class ParameterSet(object):
    """
    A parameteter set which is a container for a Parameters objects and a Dimensions objects.
    """

    def __init__(self):
        """Create a new ParameterSet"""

        self.__parameters = Parameters()
        self.__dimensions = Dimensions()

    @property
    def dimensions(self):
        """Returns the Dimensions object"""
        return self.__dimensions

    @property
    def parameters(self):
        """Returns the Parameters object"""
        return self.__parameters

    @property
    def xml_global_dimensions(self):
        """Return an xml ElementTree of the dimensions used by all parameters"""

        dims_xml = xmlET.Element('dimensions')

        for kk in self.dimensions.keys():
            dim_sub = xmlET.SubElement(dims_xml, 'dimension')
            dim_sub.set('name', kk)
        return dims_xml

    @property
    def xml_global_parameters(self):
        """Return an xml ElementTree of the parameters"""
        inv_map = {vv: kk for kk, vv in iteritems(NHM_DATATYPES)}
        # print(inv_map)

        params_xml = xmlET.Element('parameters')

        for vv in self.parameters.values():
            # print(vv.name, inv_map[vv.datatype])
            param_sub = xmlET.SubElement(params_xml, 'parameter')
            param_sub.set('name', vv.name)
            param_sub.set('type', inv_map[vv.datatype])

            if vv.model:
                param_sub.set('model', vv.model)
            if vv.description:
                param_sub.set('desc', vv.description)
            if vv.help:
                param_sub.set('help', vv.help)

        return params_xml

    def _read(self):
        assert False, 'ParameterSet._read() must be defined by child class'

    def write_parameters_xml(self, output_dir):
        """Write global parameters.xml"""
        # Write the global parameters xml file
        xmlstr = minidom.parseString(xmlET.tostring(self.xml_global_parameters)).toprettyxml(indent='    ')
        with open('{}/{}'.format(output_dir, PARAMETERS_XML), 'w') as ff:
            ff.write(xmlstr)

    def write_dimensions_xml(self, output_dir):
        """Write global dimensions.xml"""
        # Write the global dimensions xml file
        xmlstr = minidom.parseString(xmlET.tostring(self.xml_global_dimensions)).toprettyxml(indent='    ')
        with open('{}/{}'.format(output_dir, DIMENSIONS_XML), 'w') as ff:
            ff.write(xmlstr)