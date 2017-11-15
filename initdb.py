"""
Database Initializer
"""

SCHEMA_RECEIVED = \
'''CREATE TABLE `received` (
        `datetime`              INTEGER UNIQUE,
        `offset_ms`             INTEGER,
        `freq_khz`              INTEGER,
        `bfo_offset_hz`         INTEGER,
        `recorder`              TEXT,
        `char1_max_sn`          REAL,
        `char1_best_pos_hz`     INTEGER,
        `char1_total_ct`        INTEGER,
        `char1_bg_pos_hz`       INTEGER,
        `char1_bg_sn`           REAL,
        `bayes1_prob`           REAL,
        PRIMARY KEY(`datetime`)
)'''

from lib.common import eprint
from lib.fileio import connect_database

def init_db(destroy='no', preserve=False, debug=False):
    from lib.config import BeaconConfigParser
    import os
    if destroy != 'yes':
        raise Exception('Not accepted by "yes"')

    if not preserve:
        try:
            os.remove(BeaconConfigParser().get('Common', 'database'))
        except OSError as err:
            if err[1] != 'No such file or directory':
                raise

    conn = connect_database()
    c = conn.cursor()
    c.execute(SCHEMA_RECEIVED)
    conn.commit()
    conn.close()
    eprint('Database is initialized and set up.')

def main():
    import argparse
    import re
    import sys

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Initialize (clear) database')
    parser.add_argument('-d', '--debug',
        action='store_true',
        default=False,
        help='enable debug')
    parser.add_argument('--preserve',
        action='store_true',
        default=False,
        help='do not erase database file')
    parser.add_argument('agree',
        help='say "agree" to initialize database')
    args = parser.parse_args()

    # Check arguments
    m = re.match(r'agree$', args.agree)
    if not m:
        eprint('usage: "python initdb.py agree" to initialize (clear) database')
        sys.exit(1)

    # Ask again because can't undo
    eprint("Final confirmation: database will be destroyed.")
    s = raw_input("It's unrecoverable.  Are you sure? (yes or no): ")

    if s == 'yes':
        init_db(destroy='yes', preserve=args.preserve, debug=args.debug)
    else:
        eprint('Aborted.  (Not initialized.)')

if __name__ == "__main__":
    main()
