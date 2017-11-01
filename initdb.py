"""
Database Initializer
"""

from lib.common import eprint
from lib.fileio import connect_database

def init_db(debug=False):
    con = connect_database()
    eprint('INITIALIZED')

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
        init_db(args.debug)
    else:
        eprint('Aborted.  (Not initialized.)')

if __name__ == "__main__":
    main()
