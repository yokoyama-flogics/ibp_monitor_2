"""
File Handlers

BSD 2-Clause License

Copyright (c) 2017, Atsushi Yokoyama, Firmlogics (yokoyama@flogics.com)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
