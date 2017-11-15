"""
Configuration File Parser
"""

import ConfigParser
import os

class BeaconConfigParser():
    def __init__(self, configfile_name=None):
        self.config = ConfigParser.RawConfigParser()
        if configfile_name:
            self.config.read(configfile_name)
        else:
            self.config.read(['bm2.cfg', os.path.expanduser('~/bm2.cfg')])

    def get(self, section, name):
        return self.config.get(section, name)

    def getint(self, section, name):
        return self.config.getint(section, name)

    def getpath(self, section, name):
        return os.path.expanduser(self.config.get(section, name))
