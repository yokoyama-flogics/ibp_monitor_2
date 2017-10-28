"""
File Handlers
"""

from lib.config import BeaconConfigParser
import unittest

class UnitTests(unittest.TestCase):
    pass

def open_db_file(name, mode=None):
    config = BeaconConfigParser()
    dbdir = config.get('Migration', 'dbdir')
    if dbdir[-1] != '/':
        dbdir += '/'
    return open(dbdir + name, mode)

def main():
    pass

if __name__ == "__main__":
    # unittest.main()
    main()
