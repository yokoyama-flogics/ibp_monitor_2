"""
Signal Retriever

This is required to migrate from Monitor 1 to Monitor 2.
Scan signal raw file (one file a day) and generate a ten-seconds .wav file.
"""

# XXX We should take care of iqlag (I/Q sample delay in input) in the tool.

global debug

import unittest

class UnitTests(unittest.TestCase):
    pass

def retrieve_signal(date_str, line_num):
    """
    Open a Monitor 1 style '.txt' and '.raw' file and retrieve a 10-seconds
    signal data which is specified by line_num (line of the '.txt' file).
    """
    from lib.fileio import open_db_file
    import math
    import re

    txtfile = open_db_file('ibprec_' + date_str + '.txt', mode='r')

    # Dummy read (or skip) of 'line_num - 1' lines
    for i in range(line_num - 1):
        txtfile.readline()

    # Read corresponding index lines
    # (Note that consecutive 2 index lines are required.)
    cur_index = txtfile.readline()
    next_index = txtfile.readline()

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

def write_wav_file(filename, data):
    """
    Convert raw signal data 'data' to .wav format and write to 'filename'.
    """
    from lib.config import BeaconConfigParser
    import wave

    # Read parameter samplerate from config file
    config = BeaconConfigParser()
    samplerate = config.getint('Signal', 'samplerate')

    wavfile = wave.open(filename, 'wb')
    wavfile.setnchannels(2)
    wavfile.setsampwidth(2)
    wavfile.setframerate(samplerate)
    wavfile.writeframesraw(data)
    wavfile.close()

def main():
    global debug
    import argparse
    from lib.config import BeaconConfigParser
    import re

    # Parse configuration file
    config = BeaconConfigParser()

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Retrieve a 10-seconds .wav file')
    parser.add_argument('-d', '--debug',
        action='store_true',
        default=False,
        help='enable debug')
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
        print "Illegal date '%s' specified" % (args.date)
    if args.line < 1:
        print "Illegal line '%d' specified" % (args.line)
    debug = args.debug

    # Read signal data from raw file, and write it as .wav file
    sig = retrieve_signal(args.date, args.line)
    write_wav_file(args.output_file, sig)

if __name__ == "__main__":
    main()
