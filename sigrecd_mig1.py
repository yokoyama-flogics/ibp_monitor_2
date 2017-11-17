"""
Signal Recorder Daemon for Migration from IBP Monitor-1
"""

import os
import sys

LEN_INPUT_SEC = 10      # length of sigdata must be 10 seconds signal

# Set Python search path to the parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.common import eprint

def exceeded_sigfiles_limit():
    """
    Check if too many signal files are generated and return True if so.
    """
    from lib.config import BeaconConfigParser
    import re

    prog = re.compile(r'\.wav$')

    ct = 0
    for root, dirs, files in \
            os.walk(BeaconConfigParser().getpath('Signal', 'dir')):
        for f in files:
            if prog.search(f):
                ct += 1

    if ct >= BeaconConfigParser().getint(
            'SignalRecorder', 'sigfiles_num_limit'):
        return True
    else:
        return False

def record_one_file(datestr, timestr, line, skip_if_exist=False, debug=False):
    """
    Record (or 'convert' in the migration recorder case) one file from
    the raw file specified by 'datestr' and 'line' in the file.
    Note that the 'line' is true line number of the file.  Comment line is also
    counted.
    """
    from lib.config import BeaconConfigParser
    from lib.fileio import getpath_signalfile
    from sigretr import retrieve_signal, write_wav_file, adjust_len
    import os
    import wave

    if not hasattr(record_one_file, 'n_samples'):
        record_one_file.n_samples = \
            LEN_INPUT_SEC * BeaconConfigParser().getint('Signal', 'samplerate')

    filename = datestr + '/' + timestr + '.wav'
    filepath = getpath_signalfile(datestr + '/' + timestr + '.wav')
    # print filepath

    # If the signal file exists and can be ignored, skip file retrieval
    try:
        if skip_if_exist and \
                wave.open(filepath, 'rb').getnframes() == \
                record_one_file.n_samples:
            return
    except IOError as err:
        if err[1] == 'No such file or directory':
            # File does not exist...
            pass
        else:
            raise
    except:
        raise

    # Read signal data from raw file, and write it as .wav file
    sig = retrieve_signal(datestr, line, debug=False)
    sig = adjust_len(sig)
    write_wav_file(filename, sig, to_signal_dir=True)

def nextday_datestr(datestr):
    """
    Return the next day's datestr of the given datestr.
    20171101 will be returned if 20171031 was given, for example.
    """
    from datetime import datetime, timedelta

    nextday = datetime.strptime(datestr, '%Y%m%d') + timedelta(days=1)
    return nextday.strftime('%Y%m%d')

def register_db(datestr, timestr, mhz, ignore_err=False, debug=False):
    from datetime import datetime
    from lib.config import BeaconConfigParser
    from lib.fileio import connect_database
    from lib.ibp import mhz_to_freq_khz
    from sqlite3 import IntegrityError

    # Convert datestr and timestr to seconds from epoch
    datetime_utc = datetime.strptime(
        datestr + ' ' + timestr, '%Y%m%d %H%M%S')
    seconds_from_epoch = int(
        (datetime_utc - datetime.utcfromtimestamp(0)).total_seconds())
    if debug:
        print "seconds_from_epoch:", seconds_from_epoch
    if seconds_from_epoch % 10 != 0:
        raise Exception('seconds_from_epoch is not multiple of 10 seconds')

    # Obtain parameters from configuration
    config = BeaconConfigParser()

    conn = connect_database()
    c = conn.cursor()
    try:
        c.execute('''INSERT INTO
            received(datetime, offset_ms, freq_khz, bfo_offset_hz, recorder)
            VALUES(?,?,?,?,?)''',
            (
                seconds_from_epoch,
                config.getint('Migration', 'offset_ms'),
                mhz_to_freq_khz(mhz),
                config.getint('Migration', 'bfo_offset_hz'),
                config.get('Migration', 'recorder')
            ))
        conn.commit()
    except IntegrityError as err:
        if ignore_err and \
                err[0] == 'UNIQUE constraint failed: received.datetime':
            pass
        else:
            raise

    conn.close()

def startrec(arg_from, ignore_err=False, check_limit=False, debug=False):
    """
    Repeat signal conversion from 'arg_from'.
    If arg_from is 'new', it start from new line of today's file.
    Otherwise, treats arg_from as datestr (e.g. 20171028)
    """
    from datetime import datetime
    from lib.fileio import open_db_file
    import math
    import re
    import time

    if arg_from == 'new':
        datestr = datetime.utcnow().strftime('%Y%m%d')
        seek_to_tail = True
    elif arg_from == 'today':
        datestr = datetime.utcnow().strftime('%Y%m%d')
        seek_to_tail = False
    else:
        datestr = arg_from
        seek_to_tail = False

    curfd = None
    curline = 0     # will be initialized in the loop below in anyway

    while True:
        # For first iteration or curfd is closed, open a new file
        if curfd is None:
            curfd = open_db_file('ibprec_%s.txt' % (datestr), 'r')
            curline = 0

        # If seek_to_tail is True, it means starting from new line (next of the
        # current last line, so skip already existing lines.
        if seek_to_tail == True:
            seek_to_tail = False    # never seek again
            while True:
                line = curfd.readline()
                if line == '':
                    break       # no more lines

                curline += 1
                if debug:
                    print "#", line.rstrip()    # skipped line

        # Now, wait for a line from curfd
        while True:
            line = curfd.readline()
            if line == '':
                time.sleep(0.5)     # sleep for a short period
            else:
                curline += 1
                break

        line = line.rstrip()

        # Check if the line is a comment-only line
        m = re.match(r' *#', line)
        if m:
            if debug:
                print "COMMENT:", line
            continue

        # Check if the line is an end-of-file marker
        m = re.search(r'MHz\s*$', line)
        if not m:
            if debug:
                print "TERMINATED:", line
            curfd.close()
            curfd = None
            curline = 0
            datestr = nextday_datestr(datestr)
            continue

        if debug:
            print "%4d: %s" % (curline, line)

        # This is not required.  %H:%M:%S is included in the line...
        # m = re.match(r'\d+', line)
        # utctime = datetime.utcfromtimestamp(
        #     math.floor((float(m.group(0)) + 6.0) / 10.0) * 10.0)

        # Extract time time string from the line, and convert to %H%M%S.
        m = re.search(r' (\d{2}):(\d{2}):(\d{2}) ', line)
        timestr = m.group(1) + m.group(2) + m.group(3)

        # Extract frequency (kHz) from the line
        m = re.search(r' (\d+)MHz\s*$', line)
        mhz = int(m.group(1))

        if check_limit:
            # Check if generated signal files are too much
            if exceeded_sigfiles_limit():
                eprint("Signal files limit (number of size) is exceeded.")
                eprint("Waiting until not meeting the condition again.")
            while exceeded_sigfiles_limit():
                time.sleep(0.5)

        # Finally process the line
        record_one_file(datestr, timestr, curline, ignore_err, debug)
        register_db(datestr, timestr, mhz, ignore_err, debug=debug)

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
    parser.add_argument('--checklimit',
        action='store_true',
        default=False,
        help='check if generated signal files are too many (it makes very slow')
    parser.add_argument('--force',
        action='store_true',
        default=False,
        help='ignore error even record and signal files already exist in' +
            ' database or directory')
    parser.add_argument('-f', '--from',
        # required=True,
        help='process from "new" (default), "today" (00:00:00 in UTC),'
            ' or datestr (e.g. 20171028)')
    parser.add_argument('--daemon',
        # nargs=1,
        choices=['start', 'stop', 'restart'],
        help='run as daemon.  start, stop, or restart')
    args = parser.parse_args()

    args.arg_from = getattr(args, 'from')

    # Check arguments
    if args.arg_from == 'new':
        pass
    elif args.arg_from == 'today':
        pass
    elif args.arg_from is None:
        args.arg_from = 'new'
    else:
        m = re.match(r'[0-9]{8}$', args.arg_from)
        if not m:
            eprint("Illegal datestr '%s' specified" % (args.arg_from))
            sys.exit(1)

    startrec(args.arg_from, ignore_err=args.force, check_limit=args.checklimit,
        debug=args.debug)

if __name__ == "__main__":
    main()
