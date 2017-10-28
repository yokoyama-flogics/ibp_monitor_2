"""
Configuration File Parser
"""

import ConfigParser
import os

class BeaconConfigParser():
    def __init__(self):
        self.config = ConfigParser.RawConfigParser()
        self.config.read(['bm2.cfg', os.path.expanduser('~/bm2.cfg')])

    def get(self, section, name):
        return self.config.get(section, name)

    def getint(self, section, name):
        return self.config.getint(section, name)
