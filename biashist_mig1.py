"""
Bias History Migration Utility
"""

import os
import sys

# Set Python search path to the parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.common import eprint

def biashist_mig_band(dbconn, band, recorder, filename, ignore_err=False):
    """
    Read lines from given filename (Monitor-1 biashist file) and insert them as
    database records.
    """
    import re
    import sqlite3
    from lib.ibp import callsign_to_slot, get_slot

    for line in open(filename, 'r').readlines():
        m = re.match('(\d+) .*SN: *([\d.-]+) Bias: *([\d.-]+) Ct: *(\d+)', line)
        datetime_sec = (int(m.group(1)) + 6) / 10 * 10
        sn = float(m.group(2))
        bias_hz = int(m.group(3))
        ct = int(m.group(4))
        # print datetime_sec, sn, bias_hz, ct

        m = re.search(r'_([A-Z0-9]+)_', filename)
        callsign = m.group(1)
        bad_slot = get_slot(datetime_sec, band)
        true_slot = callsign_to_slot(callsign)
        diff = (bad_slot - true_slot) % 18
        if diff < 2 or diff > 3:
            # print bad_slot, callsign
            print diff

        c = dbconn.cursor()
        try:
            c.execute('''INSERT INTO
                biashist(datetime, band, recorder, sn, bias_hz, ct)
                VALUES(?,?,?,?,?,?)''',
                (
                    datetime_sec,
                    band,
                    recorder,
                    sn,
                    bias_hz,
                    ct
                ))
            dbconn.commit()
        except sqlite3.IntegrityError as err:
            if not ignore_err:
                raise
            elif err[0] != 'UNIQUE constraint failed: biashist.datetime':
                raise

def biashist_mig_all(ignore_err=False, debug=False):
    from lib.config import BeaconConfigParser
    from lib.fileio import connect_database
    from fnmatch import fnmatch
    import os

    dbdir = BeaconConfigParser().get('Migration', 'dbdir')
    recorder = BeaconConfigParser().get('Migration', 'recorder')
    conn = connect_database()

    for band in (14, 18, 21, 24, 28):
        for file in os.listdir(dbdir):
            if fnmatch(file, 'ibprec_*_%dMHz.hist' % (band)):
                if debug:
                    print "Migrating", file
                biashist_mig_band(conn, band, recorder,
                    os.path.join(dbdir, file), ignore_err=ignore_err)

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
    parser.add_argument('--ignoreerr',
        action='store_true',
        default=False,
        help='continue even error occurred when inserting records')
    args = parser.parse_args()

    biashist_mig_all(ignore_err=args.ignoreerr, debug=args.debug)

if __name__ == "__main__":
    main()
