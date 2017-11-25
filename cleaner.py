"""
Signal Files Cleaner

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
    import time

    while True:
        cleanup(debug=False)
        time.sleep(60)

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
