"""
Signal Recorder Daemon for Migration from IBP Monitor-1
"""

from lib.common import eprint
from lib.fileio import open_db_file

def nextday_datestr(datestr):
    """
    Return the next day's datestr of the given datestr.
    20171101 will be returned if 20171031 was given, for example.
    """
    from datetime import datetime, timedelta

    nextday = datetime.strptime(datestr, '%Y%m%d') + timedelta(days=1)
    return nextday.strftime('%Y%m%d')

def startrec(arg_from, debug=False):
    """
    Repeat signal conversion from 'arg_from'.
    If arg_from is 'new', it start from new line of today's file.
    Otherwise, treats arg_from as datestr (e.g. 20171028)
    """
    import re
    import time

    if arg_from == 'new':
        from datetime import datetime
        datestr = datetime.utcnow().strftime('%Y%m%d')
        start_line = -1
    else:
        datestr = arg_from
        start_line = 1

    curfd = None

    while True:
        # For first iteration or curfd is closed, open a new file
        if curfd is None:
            curfd = open_db_file('ibprec_%s.txt' % (datestr), 'r')

        # If start_line < 0, if means starting from new line, so skip already
        # existing lines.
        if start_line < 0:
            while True:
                line = curfd.readline()
                if line == '':
                    break
                if debug:
                    print "* ", line.rstrip()    # skipped line

        # Now, wait for a line from curfd
        while True:
            line = curfd.readline()
            if line == '':
                time.sleep(0.5)     # sleep for a short period
            else:
                break

        line = line.rstrip()

        # Check if the line is a comment-only line
        m = re.match(r' *#', line)
        if m:
            if debug:
                print "COMMENT: ", line
            continue

        # Check if the line is an end-of-file marker
        m = re.search(r'MHz\s*$', line)
        if not m:
            if debug:
                print "TERMINATED: ", line
            curfd = None
            datestr = nextday_datestr(datestr)
            start_line = 1

        if debug:
            print "# ", line

def main():
    import argparse
    import re
    import sys

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Signal Recorder Daemon for Migration from IBP Monitor-1')
    parser.add_argument('-d', '--debug',
        action='store_true',
        default=False,
        help='enable debug')
    parser.add_argument('-f', '--from',
        # nargs=1,
        help='process from "new" (default), or datestr (e.g. 20171028)')
    parser.add_argument('--daemon',
        # nargs=1,
        choices=['start', 'stop', 'restart'],
        help='run as daemon.  start, stop, or restart')
    args = parser.parse_args()

    args.arg_from = getattr(args, 'from')

    # Check arguments
    if args.arg_from == 'new':
        pass
    else:
        m = re.match(r'[0-9]{8}$', args.arg_from)
        if not m:
            eprint("Illegal datestr '%s' specified" % (args.arg_from))
            sys.exit(1)

    startrec(args.arg_from, debug=args.debug)

if __name__ == "__main__":
    main()
