"""
Bias History Migration Utility
"""

import os
import sys

# Set Python search path to the parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.common import eprint

def biashist_mig_band(dbconn, recorder, offset_ms, bfo_offset_hz, filename,
        ignore_err=False):
    """
    Read lines from given filename (Monitor-1 biashist file) and insert them as
    database records.
    """
    from lib.config import BeaconConfigParser
    from lib.ibp import mhz_to_freq_khz
    import re
    import sqlite3
    from datetime import datetime

    m = re.search('_(20[0-9]+)\.log', filename)
    date_str = m.group(1)

    for line in open(filename, 'r').readlines():
        if line.rstrip() == 'END':
            break

        if line.rstrip() == '':
            eprint('Found empty line.  Skipped')
            continue

        # Parsing characteristic parameters from *.log file
        m = re.match(
            '([0-9:]+) [A-Z0-9]+ +(\d+)MHz SN: *([\d.-]+) Bias: *([\d.-]+)'
            + ' Ct: *(\d+) IF: *([\d-]+) +([\d.-]+)',
            line)
        try:
            datetime_sec = (datetime.strptime(
                date_str + ' ' + m.group(1),
                '%Y%m%d %H:%M:%S')
                - datetime.utcfromtimestamp(0)).total_seconds()
        except:
            eprint('Found illegal line "%s".  Aborted')
            raise

        freq_khz = mhz_to_freq_khz(int(m.group(2)))
        max_sn = float(m.group(3))
        best_pos_hz = int(m.group(4))
        total_ct = int(m.group(5))
        bg_pos_hz = int(m.group(6))
        bg_sn = float(m.group(7))
        # print datetime_sec, freq_khz, max_sn, best_pos_hz, total_ct
        # print bg_pos_hz, bg_sn

        # Originally, trying to calculate true time by comparing bad_slot and
        # true slot.
        # m = re.search(r'_([A-Z0-9]+)_', filename)
        # callsign = m.group(1)
        # bad_slot = get_slot(datetime_sec, band)
        # true_slot = callsign_to_slot(callsign)
        # diff = (bad_slot - true_slot) % 18
        # if diff < 2 or diff > 3:
        #     # print bad_slot, callsign
        #     print diff

        c = dbconn.cursor()
        try:
            c.execute('''INSERT INTO
                received(datetime, offset_ms, freq_khz, bfo_offset_hz, recorder,
                char1_max_sn, char1_best_pos_hz, char1_total_ct,
                char1_bg_pos_hz, char1_bg_sn)

                VALUES(?,?,?,?,?,?,?,?,?,?)''',
                (
                    datetime_sec,
                    offset_ms,
                    freq_khz,
                    bfo_offset_hz,
                    recorder,
                    max_sn,
                    best_pos_hz,
                    total_ct,
                    bg_pos_hz,
                    bg_sn
                ))
        except sqlite3.IntegrityError as err:
            if not ignore_err:
                raise
            elif err[0] != 'UNIQUE constraint failed: biashist.datetime':
                raise

    dbconn.commit()

def biashist_mig_all(ignore_err=False, debug=False):
    from lib.config import BeaconConfigParser
    from lib.fileio import connect_database
    from fnmatch import fnmatch
    import os

    dbdir = BeaconConfigParser().getpath('Migration', 'dbdir')
    recorder = BeaconConfigParser().get('Migration', 'recorder')
    offset_ms = BeaconConfigParser().getint('Migration', 'offset_ms')
    bfo_offset_hz = \
        BeaconConfigParser().getint('Migration', 'bfo_offset_hz')

    conn = connect_database()

    for file in sorted(os.listdir(dbdir)):
        if fnmatch(file, 'ibprec_*.log'):
            if debug:
                print "Migrating", file
            biashist_mig_band(conn, recorder, offset_ms, bfo_offset_hz,
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
