"""
Signal Files Cleaner
"""

import os
import sys

# Set Python search path to the parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.common import eprint

def cleanup(debug=False):
    """
    Search signal files and remove it if it's old
    """
    from datetime import datetime
    from lib.config import BeaconConfigParser
    from lib.fileio import connect_database
    import os
    import re

    config = BeaconConfigParser()
    files_path = config.getpath('Signal', 'dir')
    timelimit_sec = eval(config.get('Cleaner', 'timelimit_sec'))
    if debug:
        print 'timelimit_sec = %d' % (timelimit_sec)

    conn = connect_database()
    c = conn.cursor()

    for date_dir in os.listdir(files_path):
        if not re.match(r'[0-9]{8}$', date_dir):
            continue

        # Now found a date directory

        date_dir_path = os.path.join(files_path, date_dir)
        for file in os.listdir(date_dir_path):
            m = re.match(r'([0-9]{6})\.wav$', file)
            if not m:
                continue

            time_str = m.group(1)

            datetime_sec = int((datetime.strptime(
                date_dir + ' ' + time_str,
                '%Y%m%d %H%M%S')
                - datetime.utcfromtimestamp(0)).total_seconds())
            # print date_dir, time_str, datetime_sec

            c.execute('''SELECT datetime
                FROM received
                WHERE datetime == ? AND char1_max_sn IS NOT NULL''',
                (datetime_sec, ))

            row = c.fetchone()

            # If the signal files hasn't have characteristics in database,
            # it must be skipped.
            if row is None:
                continue

            # If the file is too old, now we can remove the signal file

            # print row
            sec_diff = int((datetime.utcnow()
                - datetime.utcfromtimestamp(0)).total_seconds()) - datetime_sec
            # print sec_diff

            if sec_diff > timelimit_sec:
                rm_filename = os.path.join(files_path, date_dir, file)
                if debug:
                    print 'Removing file %s' % (rm_filename)
                os.remove(rm_filename)

        # If the date directory is empty, remove the directory
        if os.listdir(date_dir_path) == []:
            if debug:
                print 'Removing directory %s' % (date_dir_path)
            os.rmdir(date_dir_path)

    return

def task():
    """
    Entry point for Task Keeper
    """
    cleanup(debug=False)

def main():
    import argparse
    import re
    import sys

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Signal Files Cleaner')
    parser.add_argument('-d', '--debug',
        action='store_true',
        default=False,
        help='enable debug')
    args = parser.parse_args()

    cleanup(debug=args.debug)

if __name__ == "__main__":
    main()
