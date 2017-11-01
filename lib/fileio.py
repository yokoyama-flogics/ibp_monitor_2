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
    config = BeaconConfigParser()
    dbdir = config.get('Migration', 'dbdir')
    if dbdir[-1] != '/':
        dbdir += '/'
    return open(dbdir + name, mode)

def main():
    pass

if __name__ == "__main__":
    main()
