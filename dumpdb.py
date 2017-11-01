"""
Dump Database
"""

import os
import sys

# Set Python search path to the parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.common import eprint
from lib.fileio import connect_database

def dumpdb(debug=False):
    import time

    conn = connect_database()
    c = conn.cursor()
    c.execute('''SELECT datetime, offset_ms, freq_khz, bfo_offset_hz, recorder
        FROM received
        ORDER BY datetime''')

    print 'UTC date   time     offset_ms  freq_khz  bfo_offset_hz  recorder'
    print '-' * 79

    for row in c.fetchall():
        print '%s %9d  %8d  %13d  %s' % (
            time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(row[0])),
            row[1],
            row[2],
            row[3],
            row[4])

    conn.close()

def main():
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Dump Database')
    parser.add_argument('-d', '--debug',
        action='store_true',
        default=False,
        help='enable debug')
    args = parser.parse_args()

    dumpdb(debug=args.debug)

if __name__ == "__main__":
    main()
