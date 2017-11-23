"""
File Handlers
"""

from lib.config import BeaconConfigParser
import os

def mkdir_if_required(filename):
    dirname = os.path.dirname(filename)
    if dirname != '':
        try:
            os.makedirs(dirname)
        except OSError as err:
            if err[1] != 'File exists':
                raise

def open_db_file(name, mode=None):
    dbdir = BeaconConfigParser().getpath('Migration', 'dbdir')
    return open(os.path.join(dbdir, name), mode)

def connect_database():
    import sqlite3
    database_file = BeaconConfigParser().getpath('Common', 'database')
    mkdir_if_required(database_file)
    return sqlite3.connect(database_file)

def getpath_signalfile(filename):
    """
    Return the actual path name of signal file by given filename
    """
    sigdir = BeaconConfigParser().getpath('Signal', 'dir')
    return os.path.join(sigdir, filename)

def main():
    pass

if __name__ == "__main__":
    main()
