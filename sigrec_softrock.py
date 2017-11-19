"""
Signal Recorder for SoftRock + ALSA Audio
"""

OFFSET_MS = -1000
FREQ_CHANGE_TIMING = (10000 - 1200) / 100
FILE_CHANGE_TIMING = (10000 + OFFSET_MS) / 100

import os
import sys

LEN_INPUT_SEC = 10      # length of sigdata must be 10 seconds signal   XXX

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
    counted.  And return True.
    Return false if signal file already existed.
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

    # If the signal file exists and can be ignored, skip file retrieval
    try:
        if skip_if_exist and \
                wave.open(filepath, 'rb').getnframes() == \
                record_one_file.n_samples:
            return False
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

    return True

def nextday_datestr(datestr):
    """
    Return the next day's datestr of the given datestr.
    20171101 will be returned if 20171031 was given, for example.
    """
    from datetime import datetime, timedelta

    nextday = datetime.strptime(datestr, '%Y%m%d') + timedelta(days=1)
    return nextday.strftime('%Y%m%d')

def register_db(datestr, timestr, mhz, ignore_err=False, debug=False):
    """
    Register record information to database and return True.
    Return false if a duplicate record was found.
    """
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

    err_occurred = False
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
            err_occurred = True
        else:
            raise

    conn.close()
    return not err_occurred

def old_startrec(check_limit=False, debug=False):   # XXX
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

            if debug:
                print "Skipping to tail of the file..."

            while True:
                line = curfd.readline()
                if line == '':
                    break       # no more lines

                curline += 1

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
        status1 = record_one_file(datestr, timestr, curline, ignore_err, debug)
        status2 = register_db(datestr, timestr, mhz, ignore_err, debug=debug)

        # If some processes were required, show the progress
        if status1 or status2:
            print line

def sec_x10(time):
    """
    sec_x10 returns value from 0 to 99.
    0.0 sec corresponds to 0.
    3.4 sec corresponds to 34, and so on.
    51.2 sec corresponds to 12 (not 512), and so on.
    """
    return (time.second % 10) * 10 + time.microsecond / 100000

def calc_pos(samplerate, n_samples, last_time, cur_time):
    """
    Calculate sample position correspond to the OFFSET_MS
    """
    len_time = (cur_time - last_time).total_seconds()
    truerate = n_samples / len_time
    samples_truncate = int(
        - (OFFSET_MS + last_time.microsecond / 1e3) / 1000 * truerate + 0.5)
    # print 'TRUNC=', samples_truncate
    if samples_truncate < 0:
        raise Exception

    return samples_truncate

def output_signal(datetime_sec, samples, samplerate):
    """
    Record (or 'convert' in the migration recorder case) one file from
    the raw file specified by 'datestr' and 'line' in the file.
    Note that the 'line' is true line number of the file.  Comment line is also
    counted.  And return True.
    Return false if signal file already existed.
    """
    from lib.config import BeaconConfigParser
    from lib.fileio import mkdir_if_required, getpath_signalfile
    from sigretr import retrieve_signal, write_wav_file, adjust_len
    import os
    import time
    import wave
    import numpy as np
    import sys      # XXX

    n_samples = len(samples) / 4
    np.set_printoptions(edgeitems=1000000)

    lrlag = BeaconConfigParser().getint('SignalRecorder', 'lrlag')
    filename = getpath_signalfile(
        time.strftime('%Y%m%d/%H%M%S.wav', time.gmtime(datetime_sec)))
    print filename

    # filepath = getpath_signalfile(datestr + '/' + timestr + '.wav')
    s = np.frombuffer(samples, dtype=np.dtype(np.int16))
    s = s.reshape((n_samples, 2))
    print len(s), s.shape

    ch_L = s[:, 0]
    ch_R = s[:, 1]

    # Adjust lag if required
    if lrlag > 0:
        lag = lrlag
        ch_R[0 : n_samples - lag] = ch_R[lag : n_samples]
    elif lrlag < 0:
        lag = - lrlag
        ch_L[0 : n_samples - lag] = ch_R[lag : n_samples]

    ch_I = ch_L
    ch_Q = ch_R

    out_samples = np.column_stack((ch_I, ch_Q)).flatten()
    bytes = bytearray(out_samples)

    mkdir_if_required(filename)

    wavfile = wave.open(filename, 'wb')
    wavfile.setnchannels(2)
    wavfile.setsampwidth(2)
    wavfile.setframerate(samplerate)
    wavfile.writeframesraw(bytes)
    wavfile.close()

    return True

class CutOutSamples:
    """
    Repeatably receive samples and output 10 seconds wave file when enough
    samples are collected.
    """
    def __init__(self, samplerate):
        self.samplerate = samplerate
        self.samples = bytearray('')
        self.first = True
        self.head_time = None

    def extend(self, time, samples):
        from datetime import datetime
        self.samples.extend(samples)
        # print time, len(self.samples)

        self.cur_sec_x10 = sec_x10(time)
        if self.first:
            self.last_sec_x10 = self.cur_sec_x10
            self.lasttime = time
            self.first = False
            return

        if self.last_sec_x10 < FILE_CHANGE_TIMING and \
            self.cur_sec_x10 >= FILE_CHANGE_TIMING:
                if self.head_time is not None:
                    # print 'Now output file'
                    start_sample = calc_pos(self.samplerate, \
                        len(self.samples) / 2 / 2,      # 2 ch * S16_LE
                        self.head_time,
                        time)

                    datetime_sec = int(
                        (time - datetime(1970, 1, 1)).total_seconds())
                    datetime_sec = (datetime_sec / 10) * 10

                    output_signal(datetime_sec, self.samples[
                        start_sample * 2 * 2 : \
                        (start_sample + self.samplerate * 10) * 2 * 2], \
                        # 2 ch * S16_LE
                        self.samplerate)
                    # register

                self.samples = samples
                self.head_time = self.lasttime

        self.last_sec_x10 = self.cur_sec_x10
        self.lasttime = time

from multiprocessing import Process
class SigProc(Process):
    """
    Signal Processor running on a different process (thread) so that it doesn't
    affect ALSA capturing even if SigProc is blocked by some reason
    """
    import numpy as np

    def __init__(self, samplerate, queue):
        Process.__init__(self)
        self.queue = queue
        self.samplerate = samplerate

    def run(self):
        cutout = CutOutSamples(self.samplerate)
        while True:
            try:
                new_samples, now = self.queue.get()
                cutout.extend(now, bytearray(new_samples))
            except KeyboardInterrupt:
                eprint('Interrupted by user.  Aborted.')
                break

def change_freq(time, bfo_offset_hz, debug):
    """
    Tune receiver to the next receiving frequency
    Note that frequency for the next coming 10 sec boundary
    """
    from datetime import datetime
    from softrock import set_freq

    datetime_sec = int((time - datetime(1970, 1, 1)).total_seconds())
    datetime_sec = (datetime_sec / 10 + 1) * 10

    minute = (datetime_sec / 60) % 60
    band_idx = (minute % 15) / 3

    freq_base_khz = {
        0: 14100,
        1: 18110,
        2: 21150,
        3: 24930,
        4: 28200
    }[band_idx]

    if debug:
        print 'Changing frequency: minute=%d, freq_base_khz=%d' % \
            (minute, freq_base_khz)

    freq = freq_base_khz * 1000 - bfo_offset_hz
    set_freq(freq, debug=False)

def startrec(check_limit=False, debug=False):
    from datetime import datetime
    from lib.config import BeaconConfigParser
    from multiprocessing import Queue
    from softrock import initialize
    import alsaaudio
    import sys

    # Initalize SoftRock
    initialize(debug=debug)

    bfo_offset_hz = BeaconConfigParser().getint(
        'SignalRecorder', 'bfo_offset_hz')

    device = BeaconConfigParser().get('SignalRecorder', 'alsa_dev')
    samplerate = BeaconConfigParser().getint('Signal', 'samplerate')

    inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, device=device)
    inp.setchannels(2)

    truerate = inp.setrate(samplerate)
    if truerate != samplerate:
        eprint("Can't specify samplerate %d [Hz] to CODEC" % (samplerate))
        sys.exit(1)

    queue = Queue()
    sigproc = SigProc(samplerate, queue)
    sigproc.start()

    periodsize = samplerate / 10    # XXX  magic number
    inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)
    inp.setperiodsize(periodsize)
    
    # Continuously push the data to queue
    first = True
    while True:
        readsize, samples = inp.read()
        now = datetime.utcnow()

        # If buffer overrun occurred, tell the SigProc
        if readsize != periodsize:
            eprint('Overrun occurred.')
            samples = '\0' * periodsize * 2 * 2  # 2 ch * S16_LE

        queue.put((samples, now))

        # Change receiving frequency at appropriate timing

        cur_sec_x10 = sec_x10(now)
        if first:
            last_sec_x10 = cur_sec_x10

        if last_sec_x10 < FREQ_CHANGE_TIMING and \
                cur_sec_x10 >= FREQ_CHANGE_TIMING:
            change_freq(now, bfo_offset_hz, debug)

        # Final processes for the next iteration
        last_sec_x10 = cur_sec_x10
        first = False

    sigproc.join()

def task():
    """
    Entry point for Task Keeper
    """
    raise Exception
    startrec(arg_from='today', ignore_err=True, check_limit=False, debug=False)

def main():
    import argparse
    import re
    import sys

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Signal Recorder for SoftRock + ALSA Audio')
    parser.add_argument('-d', '--debug',
        action='store_true',
        default=False,
        help='enable debug')
    parser.add_argument('--checklimit',
        action='store_true',
        default=False,
        help='check if generated signal files are too many (it makes very slow')
    args = parser.parse_args()

    startrec(check_limit=args.checklimit, debug=args.debug)

if __name__ == "__main__":
    main()
