import sys
import unittest

sys.path.append('..')
from lib.config import *

class TestLibConfig(unittest.TestCase):
    def test_config_noconfigfile(self):
        config = BeaconConfigParser('not_exist.cfg')
        with self.assertRaises(ConfigParser.NoSectionError):
            config.get('Test', 'dbdir')

    def test_config_default(self):
        import os
        os.environ['HOME'] = 'notexist'
        config = BeaconConfigParser()
        with self.assertRaises(ConfigParser.NoSectionError):
            config.get('Signal', 'samplerate')

    def test_config_items(self):
        config = BeaconConfigParser('test_config.cfg')
        self.assertEqual(config.get('Test', 'dbdir'), 'foo')
        self.assertEqual(config.getint('Signal', 'samplerate'), 16000)

if __name__ == "__main__":
    unittest.main(buffer=True)
