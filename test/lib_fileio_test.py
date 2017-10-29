import os
import sys
import unittest

# Set Python search path to the parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.fileio import *

class TestLibFileIO(unittest.TestCase):
    def setUp(self):
        import shutil

        os.environ['HOME'] = '.'

        # Change directory to unit tests exist (Not to remove bm2.cfg in the
        # top directory)
        self.origdir = os.getcwd()
        os.chdir(os.path.join(os.path.dirname(__file__), '.'))
        shutil.copyfile('test_config.cfg', 'bm2.cfg')

    def tearDown(self):
        os.remove('bm2.cfg')
        os.chdir(self.origdir)

    def test_fileio_nodbfile(self):
        import ConfigParser
        with self.assertRaises(IOError):
            dbfile = open_db_file('not_exist.db', 'r')

    def test_fileio_dbfile(self):
        dbfile = open_db_file('test.db', 'r')
        self.assertEqual(dbfile.readline(), 'Test Database\n')

if __name__ == "__main__":
    unittest.main(buffer=True)
