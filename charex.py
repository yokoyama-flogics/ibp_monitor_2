"""
Characteristics Extractor
"""

import os
import sys

# Set Python search path to the parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.common import eprint

def read_sigdata(datetime_sec):
    """
    Read signal data (as a raw byte stream) which corresponds to the specified
    datetime_src
    """
    from lib.fileio import getpath_signalfile
    import time
    import wave

    # Open corresponding wave file
    filename = getpath_signalfile(
        time.strftime('%Y%m%d/%H%M%S.wav', time.gmtime(datetime_sec)))
    wavfile = wave.open(filename, 'rb')

    # Check properties of the signal
    if wavfile.getnchannels() != 2:
        raise Exception('Input wav file has illegal numbers of channel')
    if wavfile.getsampwidth() != 2:
        raise Exception('Input wav file has illegal sample width')

    samplerate = wavfile.getframerate()
    sigdata = wavfile.readframes(wavfile.getnframes())
    wavfile.close()

    return sigdata, samplerate

def charex(sigdata, samplerate, offset_ms, bfo_offset_hz, debug=False):
    """
    Actually calculate characteristics of the record and store them into the
    database.
    """
    import numpy as np
    import sys          # XXX

    if debug:
        eprint(samplerate, offset_ms, bfo_offset_hz)

    dtype = np.int16        # Type of each value of I or Q
    n_channels = 2          # sigdata must be L/R (I/Q) structure
    len_input_sec = 10      # Length of sigdata must be 10 seconds signal

    n_samples = samplerate * len_input_sec

    if len(sigdata) != n_samples * n_channels * np.dtype(dtype).itemsize:
        raise Exception('Length of sigdata is illegal')

    # Convert the sigdata (raw stream) to input complex vector
    iq_matrix = np.frombuffer(sigdata, dtype=dtype).reshape((n_samples, 2))
    input_vec = iq_matrix[..., 0] + 1j * iq_matrix[..., 1]
    print input_vec, len(input_vec)

    sys.exit(0)

def charex_all(debug=False):
    """
    Retrieve any record in the database, which doesn't have calculated
    characteristics by this charex.py yet, and pass them to charex()
    """
    from lib.fileio import connect_database

    conn = connect_database()
    c = conn.cursor()
    c.execute('''SELECT datetime, offset_ms, bfo_offset_hz
        FROM received
        WHERE char1_max_sn IS NULL
        ORDER BY datetime''')

    for row in c.fetchall():
        sigdata, samplerate = read_sigdata(datetime_sec = row[0])
        charex(sigdata, samplerate, row[1], row[2], debug=debug)

    conn.close()

def main():
    import argparse
    import re
    import sys

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Beacon Monitor Characteristics Extractor')
    parser.add_argument('-d', '--debug',
        action='store_true',
        default=False,
        help='enable debug')
    parser.add_argument('--daemon',
        # nargs=1,
        choices=['start', 'stop', 'restart'],
        help='run as daemon.  start, stop, or restart')
    args = parser.parse_args()

    charex_all(debug=args.debug)

if __name__ == "__main__":
    main()
