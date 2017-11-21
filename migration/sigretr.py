"""
Signal Retriever

This is required to migrate from Monitor 1 to Monitor 2.
Scans Monitor-1 style signal raw file (one file a day) and generate a
ten-seconds .wav file.
"""

from lib.common import eprint

def retrieve_signal(date_str, line_num, debug=False):
    """
    Open a Monitor-1 style '.txt' and '.raw' file and retrieve a 10-seconds
    signal data which is specified by line_num (line of the '.txt' file).
    """
    from lib.fileio import open_db_file
    import math
    import re
    import time

    txtfile = open_db_file('ibprec_' + date_str + '.txt', mode='r')

    # Dummy read (or skip) of 'line_num - 1' lines
    for i in range(line_num - 1):
        txtfile.readline()

    # Read corresponding index lines
    # (Note that consecutive 2 index lines are required.)
    cur_index = txtfile.readline()

    # Next index line may be NOT ready yet.  Need to wait in the case.
    while True:
        next_index = txtfile.readline()
        if next_index == '':
            time.sleep(0.5)
        else:
            break

    if debug:
        print "Current index: ", cur_index.rstrip()
        print "Next index: ", next_index.rstrip()

    # Extract information from the index lines
    m = re.match(r'\d+\s+(\d+)\s+(\d+)', cur_index)
    start_time = int(m.group(1))
    start_pos = int(m.group(2))
    m = re.match(r'\d+\s+(\d+)\s+(\d+)', next_index)
    end_time = int(m.group(1))
    end_pos = int(m.group(2))
    txtfile.close()

    if debug:
        print "Start:", start_time, start_pos
        print "End:", end_time, end_pos

    # end_time is actually 10 seconds ahead, so need to modify the value
    # {start,end}_time is in microseconds, so...
    end_time += 10e6
    if debug:
        print "Updated end_time: ", end_time

    n_samples = (end_pos - start_pos) / (2 * 2)     # 16-bit I/Q
    if debug:
        print "# of samples: ", n_samples

    # Next, we need to calculate 'true' starting position (in the input raw
    # file) because any line isn't located at an exact '0 usec' position.
    # Note that 'start_time' is a relative time from the '0 usec' boundary.
    true_start_pos = start_pos \
                   - (2 * 2) * int(math.floor(      # 2 * 2 due to 16-bit I/Q
                        n_samples \
                        / float(end_time - start_time) * start_time + 0.5))
    if debug:
        print "true_start_pos: ", true_start_pos

    # Finally open the raw data file, seek to true_start_pos, and return the
    # data
    rawfile = open_db_file('ibprec_' + date_str + '.raw', mode='rb')
    rawfile.seek(true_start_pos, 0)

    rawdata = rawfile.read(n_samples * (2 * 2))
    rawfile.close()
    return rawdata

def write_wav_file(filename, data, to_signal_dir=False):
    """
    Convert raw signal data 'data' to .wav format and write to 'filename'.

    If to_signal_dir is True, written to the standard direcotry.
    """
    from lib.config import BeaconConfigParser
    from lib.fileio import mkdir_if_required, getpath_signalfile
    import os
    import wave

    # First, we need to take care of PCM2902 lag between L and R channels
    iqlag = BeaconConfigParser().getint('Migration', 'iqlag')
    if iqlag > 0:
            i_ch_lag = iqlag
            q_ch_lag = 0
    else:
            i_ch_lag = 0
            q_ch_lag = -iqlag

    n_samples = len(data) / 4   # I_low, I_high, Q_low, Q_high, ...

    # Take account lag and split to I/Q
    def limit(val, maxval):
        if val > maxval:
            val = maxval
        return val

    def substr(s, start, len):
        return s[start:start + len]

    i_ch = {}
    q_ch = {}
    for i in range(n_samples):
        i_pos = limit(i + i_ch_lag, n_samples - 1)
        q_pos = limit(i + q_ch_lag, n_samples - 1)

        i_ch[i] = substr(data, ((i_pos * 2 + 0) * 2), 2)  # I comes first
        q_ch[i] = substr(data, ((q_pos * 2 + 1) * 2), 2)  # I comes first

    # Reconstruct stream from I/Q
    data = ''
    for i in range(n_samples):
        data += i_ch[i]
        data += q_ch[i]

    if to_signal_dir:
        filename = getpath_signalfile(filename)

    mkdir_if_required(filename)

    # Read parameter samplerate from config file
    samplerate = BeaconConfigParser().getint('Signal', 'samplerate')

    wavfile = wave.open(filename, 'wb')
    wavfile.setnchannels(2)
    wavfile.setsampwidth(2)
    wavfile.setframerate(samplerate)
    wavfile.writeframesraw(data)
    wavfile.close()

def adjust_len(sig):
    nominal_len = 640000
    if len(sig) < nominal_len:
        sig += chr(0) * (nominal_len - len(sig))
    elif len(sig) > nominal_len:
        sig = sig[: nominal_len]

    return sig

def main():
    import argparse
    import re
    import sys

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Retrieve a 10-seconds .wav file')
    parser.add_argument('-d', '--debug',
        action='store_true',
        default=False,
        help='enable debug')
    parser.add_argument('--exactlen',
        action='store_true',
        default=False,
        help='output exact length samples even data is short')
    parser.add_argument('date',
        help='date string e.g. 20171028')
    parser.add_argument('line',
        type=int, help='line# in the Monitor 1 .txt file')
    parser.add_argument('output_file',
        help='output .wav file name')
    args = parser.parse_args()

    # Check arguments
    m = re.match(r'[0-9]{8}$', args.date)
    if not m:
        eprint("Illegal date '%s' specified" % (args.date))
        sys.exit(1)
    if args.line < 1:
        eprint("Illegal line '%d' specified" % (args.line))
        sys.exit(1)

    # Read signal data from raw file, and write it as .wav file
    sig = retrieve_signal(args.date, args.line, debug=args.debug)

    # Adjust signal length if required
    if not args.exactlen:
        sig = adjust_len(sig)

    write_wav_file(args.output_file, sig)

if __name__ == "__main__":
    main()
