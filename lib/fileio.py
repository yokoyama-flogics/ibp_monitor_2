"""
File Handlers
"""

from lib.config import BeaconConfigParser

def mkdir_if_required(filename):
    import os
    dirname = os.path.dirname(filename)
    if dirname != '':
        try:
            os.makedirs(dirname)
        except OSError as err:
            if err[1] != 'File exists':
                raise

def open_db_file(name, mode=None):
    dbdir = BeaconConfigParser().get('Migration', 'dbdir')
    if dbdir[-1] != '/':
        dbdir += '/'
    return open(dbdir + name, mode)

def connect_database():
    import sqlite3
    database_file = BeaconConfigParser().get('Common', 'database')
    mkdir_if_required(database_file)
    print database_file
    return sqlite3.connect(database_file)

def main():
    pass

if __name__ == "__main__":
    main()
