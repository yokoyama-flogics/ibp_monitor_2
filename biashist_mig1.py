"""
Bias History Migration Utility
"""

import os
import sys

# Set Python search path to the parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.common import eprint

def biashist_mig_band(dbconn, band, filename, force=False):
    """
    Read lines from given filename (Monitor-1 biashist file) and insert them as
    database records.
    """
    import re
    import sqlite3

    for line in open(filename, 'r').readlines():
        m = re.match('(\d+) .*SN: *([\d.-]+) Bias: *([\d.-]+) Ct: *(\d+)', line)
        time_sec = (int(m.group(1)) + 6) / 10 * 10
        sn = float(m.group(2))
        bias_hz = int(m.group(3))
        ct = int(m.group(4))
        # print time_sec, sn, bias_hz, ct

        c = dbconn.cursor()
        try:
            c.execute('''INSERT INTO
                biashist(datetime, band, sn, bias_hz, ct)
                VALUES(?,?,?,?,?)''',
                (
                    time_sec,
                    band,
                    sn,
                    bias_hz,
                    ct
                ))
        except sqlite3.IntegrityError as err:
            if not force:
                raise
            if err[0] != 'UNIQUE constraint failed: biashist.datetime':
                raise

        dbconn.commit()

def biashist_mig_all(force=False, debug=False):
    from lib.config import BeaconConfigParser
    from lib.fileio import connect_database
    from fnmatch import fnmatch
    import os

    dbdir=BeaconConfigParser().get('Migration', 'dbdir')
    conn = connect_database()

    for band in (14, 18, 21, 24, 28):
        for file in os.listdir(dbdir):
            if fnmatch(file, 'ibprec_*_%dMHz.hist' % (band)):
                if debug:
                    print "Processing", file
                biashist_mig_band(conn, band, os.path.join(dbdir, file),
                    force=force)

    conn.close()

def main():
    import argparse
    import re
    import sys

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Bias History Migration Utility')
    parser.add_argument('-d', '--debug',
        action='store_true',
        default=False,
        help='enable debug')
    parser.add_argument('--force',
        action='store_true',
        default=False,
        help='update database even they already have inference')
    args = parser.parse_args()

    biashist_mig_all(force=args.force, debug=args.debug)

if __name__ == "__main__":
    main()
