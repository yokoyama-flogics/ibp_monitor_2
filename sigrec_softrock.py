"""
Signal Recorder for SoftRock + ALSA Audio

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

OFFSET_MS = -1000
FREQ_CHANGE_TIMING = (10000 - 1200) / 100
FILE_CHANGE_TIMING = (10000 + OFFSET_MS) / 100

import os
import sys

LEN_INPUT_SEC = 10      # length of sigdata must be 10 seconds signal

config = None

# Set Python search path to the parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.common import eprint

def register_db(datetime_sec):
    """
    Register record information to database and return True.
    Return false if a duplicate record was found.
    """
    from lib.fileio import connect_database

    conn = connect_database()
    c = conn.cursor()

    print 'register_db: %d' % (datetime_sec)
    c.execute('''INSERT INTO
        received(datetime, offset_ms, freq_khz, bfo_offset_hz, recorder)
        VALUES(?,?,?,?,?)''',
        (
            datetime_sec,
            OFFSET_MS,
            datetime_sec_to_freq_khz(datetime_sec),
            config.getint('SignalRecorder', 'bfo_offset_hz'),
            config.get('SignalRecorder', 'recorder')
        ))

    conn.commit()
    conn.close()

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
    from lib.fileio import mkdir_if_required, getpath_signalfile
    import os
    import time
    import wave
    import numpy as np
    import sys      # XXX

    # If length of samples are short, append zeros at the tail
    expected_n_samples = samplerate * LEN_INPUT_SEC * 2 * 2     # 2 ch * S16_LE
    if len(samples) < expected_n_samples:
        samples.extend([0] * (expected_n_samples - len(samples)))

    n_samples = len(samples) / 4
    np.set_printoptions(edgeitems=1000000)

    lrlag = config.getint('SignalRecorder', 'lrlag')
    sig_iq = config.get('SignalRecorder', 'sig_iq')

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
        ch_L[0 : n_samples - lag] = ch_L[lag : n_samples]

    # XXX   L/R from 12:33 JST Nov/20
    # XXX   R/L from 12:58 JST Nov/20 Lite9 good
    # XXX   L/R from 13:53 JST Nov/20 Lite9 bad
    # XXX   R/L from 14:56 JST Nov/20 with Ensemble III and back antenna: bad
    # XXX   R/L from 15:30 JST Nov/20 with Ensemble III and main antenna: good
    # XXX   R/L from 15:40 JST Nov/20 with Ensemble III and back antenna: bad
    # XXX   R/L from 16:18 JST Nov/20 with Ensemble III and main antenna:
    # ch_I = ch_R     # XXX   L/R from 12:33 JST Nov/20
    # ch_Q = ch_L     # XXX

    if sig_iq == 'L/R':
        ch_I = ch_L
        ch_Q = ch_R
    elif sig_iq == 'R/L':
        ch_I = ch_R
        ch_Q = ch_L
    else:
        eprint('[SignalRecorder] sig_iq must be L/R or R/L')
        raise Exception

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

                    # Not rounding up
                    datetime_sec = int(
                        (time - datetime(1970, 1, 1)).total_seconds())
                    datetime_sec = (datetime_sec / 10) * 10

                    # 2 ch * S16_LE
                    truncated_samples = self.samples[
                        start_sample * 2 * 2 : \
                        (start_sample + self.samplerate * 10) * 2 * 2]
                    if len(truncated_samples) != self.samplerate * 10 * 2 * 2:
                        print '# illegal len', len(self.samples), len(truncated_samples), start_sample, self.head_time, time   # XXX
                    output_signal(datetime_sec, truncated_samples,
                        self.samplerate)
                    register_db(datetime_sec)

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

def roundup_time_10sec(time):
    from datetime import datetime

    datetime_sec = int((time - datetime(1970, 1, 1)).total_seconds())
    return (datetime_sec / 10 + 1) * 10

def datetime_sec_to_freq_khz(datetime_sec, debug=False):
    from lib.ibp import mhz_to_freq_khz
    minute = (datetime_sec / 60) % 60
    band_idx = (minute % 15) / 3

    if debug:
        print 'datetime_sec_to_freq_khz: minute=%d' % (minute)

    return mhz_to_freq_khz({
        0: 14,
        1: 18,
        2: 21,
        3: 24,
        4: 28
    }[band_idx])    # XXX for debug

def change_freq(time, bfo_offset_hz, debug):
    """
    Tune receiver to the next receiving frequency
    Note that frequency for the next coming 10 sec boundary
    """
    from softrock import set_freq

    freq_base_khz = datetime_sec_to_freq_khz(roundup_time_10sec(time), debug)

    if debug:
        print 'Changing frequency: %d kHz' % (freq_base_khz)

    freq = freq_base_khz * 1000 - bfo_offset_hz
    set_freq(freq, debug=False)

def startrec(check_limit=False, debug=False):
    from datetime import datetime
    from multiprocessing import Queue
    from softrock import initialize
    import alsaaudio
    import sys

    # Initalize SoftRock
    initialize(debug=debug)

    bfo_offset_hz = config.getint('SignalRecorder', 'bfo_offset_hz')

    device = config.get('SignalRecorder', 'alsa_dev')
    samplerate = config.getint('Signal', 'samplerate')

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
    startrec(check_limit=False, debug=False)

def startrec_with_recover(check_limit=False, debug=False):
    """
    Even startrec() failed, it will be relaunched
    """
    global config

    from time import sleep
    import logging
    from lib.config import BeaconConfigParser

    config = BeaconConfigParser()

    logging.basicConfig(filename='sigrec.log')

    def datestr():
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    while True:
        try:
            startrec(check_limit=check_limit, debug=debug)
            eprint('startrec() exited at %s.  Continued.' % (datestr()))

        except KeyboardInterrupt:
            eprint('Interrupted by user.  Aborted.')
            break

        except:
            eprint('startrec() raised an exception at %s.  Continued.' % \
                (datestr()))
            logging.exception('startrec() at ' + datestr())

        sleep(1)

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

    startrec_with_recover(check_limit=args.checklimit, debug=args.debug)

if __name__ == "__main__":
    main()
