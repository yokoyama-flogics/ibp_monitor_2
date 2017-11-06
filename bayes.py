"""
Bayesian Inference Program
"""

import os
import sys

# Set Python search path to the parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.common import eprint

def biashist(datetime_sec, band):
    """
    Return statistical information about the beacon transmitting station.
    By specifying received time (seconds from UNIX epoch) and received band
    (14, 18, 21, and so on), check database, and return average frequency
    bias (expected exact frequency in Hz) and standard deviation.
    """
    from lib.ibp import Station

    print Station().identify_station(datetime_sec, band)

def bayes(datetime_sec, band, debug=False):
    print datetime_sec, band
    return None

def bayes_all(onepass=False, force=False, debug=False):
    """
    Retrieve any record in the database, which doesn't have Bayesian Inference
    this bayes.py yet, and pass them to bayes()
    """
    from lib.fileio import connect_database
    from lib.ibp import freq_khz_to_mhz
    import time

    conn = connect_database()
    while True:
        c = conn.cursor()

        # If specified 'force', even the record has characteristics parameters,
        # fetch any records for update.
        if force:
            cond = ''
        else:
            cond = 'WHERE bayes1_sn IS NULL'

        c.execute('''SELECT datetime, freq_khz
            FROM received
            %s
            ORDER BY datetime''' % (cond))

        for row in c.fetchall():
            paramset = bayes(row[0], freq_khz_to_mhz(row[1]), debug=debug)
            # paramset.updatedb(conn, row[0])

        if onepass:
            break
        else:
            # For continuous passes, 'force fetch' is NOT required
            force = False
            # To let rest database, wait for a short time period
            time.sleep(0.5)

    conn.close()

def main():
    import argparse
    import re
    import sys

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Bayesian Inference Program')
    parser.add_argument('-d', '--debug',
        action='store_true',
        default=False,
        help='enable debug')
    parser.add_argument('--force',
        action='store_true',
        default=False,
        help='update database even they already have inference')
    parser.add_argument('-q', '--quit',
        action='store_true',
        default=False,
        help='quit after one-pass')
    parser.add_argument('--daemon',
        # nargs=1,
        choices=['start', 'stop', 'restart'],
        help='run as daemon.  start, stop, or restart')
    args = parser.parse_args()

    bayes_all(onepass=args.quit, force=args.force, debug=args.debug)

if __name__ == "__main__":
    main()
