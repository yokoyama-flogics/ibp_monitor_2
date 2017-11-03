"""
Characteristics Extractor
"""

import os
import sys

# Set Python search path to the parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
dtype = np.int16        # type of each value of I or Q
n_channels = 2          # sigdata must be L/R (I/Q) structure
len_input_sec = 10      # length of sigdata must be 10 seconds signal
len_noise_smooth = 10   # number of samples to find neighbor range
wid_freq_detect = 500   # width of frequency (in Hz) to detect beacon
                        # (as same as detect_freq_width of Monitor-1)
sec_sg_from = 0         # stat time where beacon can be heard
                        # (definition was changed from Monitor-1)
sec_sg_until = 7        # end time where beacon can be heard
                        # (definition was changed from Monitor-1)
n_filter_order = 64
lpf_cutoff = 0.5 * 0.95
offset_sn_split = 1     # number of FFT bin to split S/N

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

def get_maxvalues_inrange(sig, len_apply):
    """
    Flatten signals by maximum values in neighbor range
    This special algorithm came from Monitor-1 proc.m.
    """
    len_sig = len(sig)
    maxvalues = np.array(sig)

    for n in range(1, len_apply):
        maxvalues = np.maximum(
            maxvalues,
            np.append(np.zeros(n), sig[0 : len_sig - n]))

    return maxvalues

def bg_est(sig, samplerate, offset_ms):
    """
    Background Estimation
    """
    from scipy.fftpack import fft

    if offset_ms > -100:    # XXX  offset_ms may be at least -100 [ms] ...
        raise Exception('Too short offset')

    # Extract very first part which shouldn't contain beacon signal
    bg_len = int((- offset_ms) / 1000.0 * samplerate)
    pre_sig = sig[0 : bg_len]

    bg = np.absolute(fft(pre_sig))
    bg_smooth = get_maxvalues_inrange(bg, len_noise_smooth)

    return bg, bg_smooth

def sg_est(sig, bg, start, samplerate, offset_ms, canceling=False):
    """
    Signal-part Estimation and Cancelling
    """
    from lib.config import BeaconConfigParser
    from scipy.fftpack import fft

    # Determine which samples should be processed
    sample_start = start - offset_ms / 1000 * samplerate
    print sample_start, sample_start + samplerate

    sg = np.absolute(fft(sig[sample_start : sample_start + samplerate]))
    sg_smooth = get_maxvalues_inrange(sg, len_noise_smooth)

    # Cancelling background if requested
    if canceling:
        true_sig = (sg_smooth / len(sg) * len(bg)) / bg
    else:
        true_sig = sg_smooth

    # Find the sig peak freq
    # It is done in the smoothed signal (background noise cancelled).
    # Now the real signal consists of +/- (sample_rate / 4) [Hz].
    # The true_sig() consists of +/- (detect_freq_width / 2) [Hz].
    bfo_offset_hz = BeaconConfigParser().getint('Migration', 'bfo_offset_hz')
    band_start = samplerate / 4 + bfo_offset_hz - wid_freq_detect / 2
    band_end   = samplerate / 4 + bfo_offset_hz + wid_freq_detect / 2
    band = true_sig[band_start : band_end]

    # Find approximate largest value and position in band
    a_pos = np.argmax(band)
    a_lvl = band[a_pos]

    # Next, obtain band_sg.  band_sg is not smoothed signal different from
    # sg_smooth
    band_sg = sg[band_start : band_end]

    # To accurately find the peak, using the not-smoothed signal but need
    # to eliminate not-likely spectrum.
    end_low_eliminate = a_pos - len_noise_smooth + 1
    if end_low_eliminate < 0:
        end_low_eliminate = 0

    begin_high_eliminate = a_pos + len_noise_smooth
    if begin_high_eliminate > len(band_sg):
        begin_high_eliminate = len(band_sg)
    len_high_eliminate = len(band_sg) - begin_high_eliminate

    band_sg[0 : end_low_eliminate] = np.zeros(end_low_eliminate)
    band_sg[begin_high_eliminate : ] = np.zeros(len_high_eliminate)
    # print a_pos, band_sg

    # Find the exact largest value and position in band
    pos = np.argmax(band_sg)
    lvl = band_sg[pos]
    # print a_pos, a_lvl
    # print pos, lvl

    # Calculating S/N.  First get 'S'
    from_sig_bin = pos - offset_sn_split
    if from_sig_bin < 0:
        from_sig_bin = 0

    to_sig_bin = pos + offset_sn_split
    if to_sig_bin >= len(band_sg):
        to_sig_bin = len(band_sg) - 1
    len_bin_sig = to_sig_bin - from_sig_bin + 1

    # print from_sig_bin, to_sig_bin
    sg_pow = np.sum(np.power(band_sg[from_sig_bin : to_sig_bin + 1], 2))

    # Then get 'N'
    band_sg[from_sig_bin : to_sig_bin + 1] = np.zeros(len_bin_sig)
    # print band_sg

    sn_db = np.log10(sg_pow / np.sum(np.power(band_sg, 2))) * 10
    # print sn_db

    return lvl, pos, sn_db

def charex(sigdata, samplerate, offset_ms, bfo_offset_hz, debug=False):
    """
    Actually calculate characteristics of the record and store them into the
    database.
    """
    from scipy import signal
    import sys          # XXX

    np.set_printoptions(edgeitems=100)   # XXX

    if debug:
        eprint(samplerate, offset_ms, bfo_offset_hz)

    n_samples = samplerate * len_input_sec

    # Generating an LPF
    # Not sure if we can apply Nuttall window and also Hamming window which
    # firwin() automatically applies.  But as same as Beacon Monitor-1 code.
    if 'lpf' not in dir(charex):
        charex.lpf = signal.nuttall(n_filter_order + 1) \
                   * signal.firwin(n_filter_order + 1, lpf_cutoff)

    # Generating an complex tone (f = samplerate / 4)
    # XXX  There are errors in latter elements but ignorable...
    if 'tone_f_4' not in dir(charex):
        charex.tone_f_4 = \
            np.exp(1j * np.deg2rad(np.arange(0, 90 * n_samples, 90)))

    if len(sigdata) != n_samples * n_channels * np.dtype(dtype).itemsize:
        raise Exception('Length of sigdata is illegal')

    # Convert the sigdata (raw stream) to input complex vector
    # It is okay that each I/Q value is 16-bit signed integer and as same as
    # the original Beacon Monitor-1 libexec/proc.m (MATLAB implementation).
    iq_matrix = np.frombuffer(sigdata, dtype=dtype).reshape((n_samples, 2))
    input_vec = iq_matrix[..., 0] + 1j * iq_matrix[..., 1]

    # input_vec is like this.
    # [ 88.-30.j  87.-29.j  88.-27.j ...,  -2. +4.j  -2. +0.j  -2. -1.j]
    # print input_vec, len(input_vec)

    # Apply LPF to narrow band width to half, and remove preceding samples
    # as same as Monitor-1
    sig = signal.lfilter(charex.lpf, 1,
        np.append(input_vec, np.zeros(n_filter_order / 2)))
    sig = sig[n_filter_order / 2 : ]

    # Applying tone (f = samplerate / 4) to shift signal upward on freq. domain
    sig *= charex.tone_f_4

    # Drop imaginary parts as same as Monitor-1
    sig = np.real(sig)
    # print sig, len(sig)

    # Background noise estimation
    bg, bg_smooth = bg_est(sig, samplerate, offset_ms)

    # Now, start analysis in signal parts of time domain
    max_sn = -np.inf
    best_pos = 0
    ct_pos = np.zeros(wid_freq_detect)

    for n in range(sec_sg_from, sec_sg_until):
        start = n * samplerate
        print '$$$', n, start
        lvl, pos, sn = \
            sg_est(sig, bg_smooth, start, samplerate, offset_ms, canceling=True)

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
