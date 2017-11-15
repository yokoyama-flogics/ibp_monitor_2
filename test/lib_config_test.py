import os
import sys
import unittest

# Set Python search path to the parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.config import *

class TestLibConfig(unittest.TestCase):
    def test_config_noconfigfile(self):
        config = BeaconConfigParser('not_exist.cfg')
        with self.assertRaises(ConfigParser.NoSectionError):
            config.getpath('Test', 'dbdir')

    def test_config_default(self):
        import os
        os.environ['HOME'] = 'notexist'
        config = BeaconConfigParser()
        with self.assertRaises(ConfigParser.NoSectionError):
            config.get('Signal', 'samplerate')

    def test_config_items(self):
        config = BeaconConfigParser('test_config.cfg')
        self.assertEqual(config.getpath('Test', 'dbdir'), 'nodb')
        self.assertEqual(config.getint('Signal', 'samplerate'), 16000)

if __name__ == "__main__":
    unittest.main(buffer=True)
