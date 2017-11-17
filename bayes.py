"""
Bayesian Inference Program
"""

VALID_THRU = 5 * 24 * 3600  # sec
MIN_TIME_WEIGHT = 0.01

import math
import os
import sys

# Set Python search path to the parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.common import eprint

class BiasHistStatistics:
    """
    BiasHist Statistics Calculator
    """
    def __init__(self):
        self.hist = {}

    def add_element(self, bin, val):
        if bin in self.hist:
            self.hist[bin] += val
        else:
            self.hist[bin] = val

    def add_params(self, passed_sec, freq_khz, comp_freq_khz, sn, bias,
            ct):
        from lib.ibp import freq_khz_to_mhz
        # print 'PP', passed_sec, freq_khz, candidate_freq_khz, sn, bias, ct

        # Calculate various weight values
        time_weight = (float(VALID_THRU) - passed_sec) / VALID_THRU
        if time_weight < MIN_TIME_WEIGHT:
            time_weight = MIN_TIME_WEIGHT

        # Convert frequencies for target and comparison data to band (14, 18,
        # ..., and so on)
        target_band = freq_khz_to_mhz(freq_khz)
        comp_band = freq_khz_to_mhz(comp_freq_khz)

        if target_band == comp_band:
            sameband_weight = 1.0
        else:
            sameband_weight = 0.3

        if math.fabs(float(bias) / comp_band) > 6.5:     # XXX  magic number
            # Received beacon frequency (or tone) looks too far from the center
            return

        if sn < 0.0:
            # S/N looks too bad
            return

        # print '#@#', passed_sec, int(float(bias) / comp_band * 28),
        #   sn, ct, time_weight, sameband_weight
        self.add_element(int(float(bias) / comp_band * 28),
            float(sn) / 10 * \
            math.pow(float(ct) / 7.0, 2) * \
            math.pow(time_weight, 2) * sameband_weight)

    def result(self):
        sum = 0.0
        weight = 0.0

        if self.hist == {}:
            # Return values as large distribution
            return float('nan'), float('nan')

        for bin in self.hist:
            sum += bin * self.hist[bin]
            weight += self.hist[bin]

        ave = sum / weight / 28

        sigma2 = 0.0
        for bin in self.hist:
            sigma2 += math.pow((float(bin) / 28 - ave), 2) * self.hist[bin]

        sigma2 /= weight

        return ave, math.pow(sigma2, 0.5)

def biashist(datetime_sec, freq_khz):
    """
    Return statistical information about the beacon transmitting station.
    By specifying received time (seconds from UNIX epoch) and received freq.
    in kHz, check database, and return average frequency bias (expected exact
    frequency in Hz) and standard deviation.
    """

    from lib.ibp import Station, freq_khz_to_mhz
    from lib.fileio import connect_database

    if not hasattr(biashist, 'identify'):
        biashist.identify = Station().identify_station

    # Identify transmitting station by time and band
    timeslot_in_sched, effective_time_sec, station = \
        biashist.identify(datetime_sec, freq_khz)
    # print '<<<', timeslot_in_sched, effective_time_sec, station

    # valid_sec is some days before the datetime_sec
    # Required not to obtain database records which are too old
    valid_sec = datetime_sec - VALID_THRU
    # print datetime_sec, valid_sec

    conn = connect_database()
    c = conn.cursor()

    # The following conditions came from Monitor-1's genhist() in
    # bin/extfeatures and also bayes/biashist
    c.execute('''SELECT datetime, freq_khz, char1_max_sn, char1_best_pos_hz,
            char1_total_ct, char1_bg_pos_hz
        FROM received
        WHERE datetime < ? AND
            datetime >= ? AND
            (char1_best_pos_hz - char1_bg_pos_hz) *
                (char1_best_pos_hz - char1_bg_pos_hz) > 4 AND
            char1_total_ct >= 3
        ORDER BY datetime''', (datetime_sec, valid_sec))

    # Search candidates and calculate statistics
    stat = BiasHistStatistics()
    for row in c.fetchall():
        candidate_datetime = row[0]
        candidate_freq_khz = row[1]
        candidate_station = \
            biashist.identify(candidate_datetime, candidate_freq_khz)
        # print '???', row, candidate_station

        # Filter stations.  Listening station and stations on the database
        # must have the same transmitting time slot and station name (or
        # transmitter).

        if timeslot_in_sched != candidate_station[0]:
            # print "$$$ different slot"
            continue

        if effective_time_sec != candidate_station[1]:
            # print "$$$ different transmitter"
            continue

        # Now found a true candidate
        passed_sec = datetime_sec - candidate_datetime
        # print '!!!', passed_sec, row, candidate_station
        sn = row[2]
        bias = row[3]
        ct = row[4]
        stat.add_params(passed_sec, freq_khz, candidate_freq_khz, sn, bias, ct)

    # print stat.hist
    return stat.result()

def stdist2(stddev, x):
    """
    I don't remember.  Related to Cumulative distribution function...
    """
    if stddev == 0.0:
        return 0.0      # to avoid division by zero

    ea = abs(x / stddev / pow(2, 0.5))

    if ea < 0.5:
        return 0.38
    elif ea < 1.5:
        return 0.24
    elif ea < 2.5:
        return 0.06
    elif ea < 3.5:
        return 0.01
    else:
        return 0.0

def dist_no(band, stddev, x):
    """
    I don't remember...
    """
    sq = pow(2, 0.5)
    s = stddev
    x = abs(x)

    # print ">>>", x, sq, s, band

    if x < 0.5 * sq * s * band:
        return 0.5 * sq * s * band / 250.0 * 2    # center has twice
    elif x < 1.5 * sq * s * band:
        return (1.5 * sq * s * band) / 250.0 - (0.5 * sq * s * band) / 250.0
    elif x < 2.5 * sq * s * band:
        return (2.5 * sq * s * band) / 250.0 - (1.5 * sq * s * band) / 250.0
    elif x < 3.5 * sq * s * band:
        return (3.5 * sq * s * band) / 250.0 - (2.5 * sq * s * band) / 250.0
    else:
        return (250.0 - (3.5 * sq * s * band)) / 250.0

class BayesInference:
    def __init__(self):
        self.ave_sn = 0.325375
        self.sigma_sn = 3.39476

        self.n_positive = 41
        self.n_total = 800
        self.pc = float(self.n_positive) / self.n_total

        C = 1   # correcting value (don't remember the reason)

        self.sn_b = {
            1: 0 + C,
            2: 0 + C,
            3: 0 + C,   # average
            4: 4,
            5: 13,
            6: 17,
            7: 6,
            8: 1
        }
        self.sn_bt = sum(self.sn_b.values())

        self.sn_n = {
            1: 0 + C,
            2: 263,
            3: 400,     # average
            4: 71,
            5: 10,
            6: 11,
            7: 3,
            8: 1
        }
        self.sn_nt = sum(self.sn_n.values())

        self.ct_b = {
            1: 0 + C,
            2: 2,
            3: 0 + C,
            4: 5,
            5: 5,
            6: 17,
            7: 12
        }
        self.ct_bt = sum(self.ct_b.values())

        self.ct_n = {
            1: 597,
            2: 97,
            3: 44,
            4: 11,
            5: 5,
            6: 5,
            7: 0 + C
        }
        self.ct_nt = sum(self.ct_n.values())

        # intentionally suppressing to add C
        self.if_b = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 41
        }
        self.if_bt = sum(self.if_b.values())

        # intentionally suppressing to add C
        self.if_n = {
            0: 20,
            1: 21,
            2: 9,
            3: 5,
            4: 2,
            5: 4,
            6: 698
        }
        self.if_nt = sum(self.if_n.values())

    def calc(self, freq_khz, bias_param, sn, ct, bias_hz, if_bias_hz):
        from lib.ibp import freq_khz_to_mhz

        band = freq_khz_to_mhz(freq_khz)

        # Calculating probability parameters (not well known)
        if math.isnan(bias_param[0]):
            off_b = 0.38
            off_n = 0.01
        else:
            ave = bias_param[0]
            stddev = bias_param[1]
            off_b = stdist2(stddev, float(bias_hz) / band - ave)
            off_n = dist_no(float(band), stddev, float(bias_hz) - ave * band)
            # print "<<<", band, off_n, bias_hz, ave, bias_param

        sn_bin = int((sn - self.ave_sn) / self.sigma_sn + (8 - 1) / 2.0)
        if sn_bin < 1:
            sn_bin = 1
        elif sn_bin > 8:
            sn_bin = 8

        diff = abs(bias_hz - if_bias_hz)
        if diff > 6:
            diff = 6

        pc = self.pc

        sn_b = self.sn_b
        sn_bt = self.sn_bt
        sn_n = self.sn_n
        sn_nt = self.sn_nt

        ct_b = self.ct_b
        ct_bt = self.ct_bt
        ct_n = self.ct_n
        ct_nt = self.ct_nt

        if_b = self.if_b
        if_bt = self.if_bt
        if_n = self.if_n
        if_nt = self.if_nt

        # print "@@@", pc, sn_b, sn_bt, ct_b, ct_bt, if_b, diff, if_bt, off_b, off_n
        # Just copied code from Monitor-1 code
        r = (pc * sn_b[sn_bin] / sn_bt * ct_b[ct] / ct_bt * if_b[diff] / if_bt * off_b) / \
                        (pc * sn_b[sn_bin] / sn_bt * ct_b[ct] / ct_bt * if_b[diff] / if_bt * off_b + \
                        ((1 - pc) * sn_n[sn_bin] / sn_nt * ct_n[ct] / ct_nt * if_n[diff] / if_nt * off_n))

        if r <= 0.0:
            r = 0.0

        return r

def bayes(bayesinf, datetime_sec, freq_khz, sn, bias_hz, ct, if_bias_hz,
        debug=False):
    """
    Bayesian Inference
    """
    if debug:
        import time

    if debug:
        print '#', datetime_sec, freq_khz, sn

    bias_param = biashist(datetime_sec, freq_khz)
    pprob = bayesinf.calc(freq_khz, bias_param, sn, ct, bias_hz, if_bias_hz)

    if debug:
        ts = time.strftime('%H:%M:%S', time.gmtime(datetime_sec))
        print ts, pprob

    return pprob

def bayes_all(onepass=False, limit=1000, force=False, debug=False):
    """
    Retrieve any record in the database, which doesn't have Bayesian Inference
    this bayes.py yet, and pass them to bayes()
    """
    from lib.fileio import connect_database
    import time

    bi = BayesInference()

    conn = connect_database()
    while True:
        c = conn.cursor()

        cond = 'WHERE char1_max_sn IS NOT NULL'

        # If specified 'force', even the record has characteristics parameters,
        # fetch any records for update.
        if not force:
            cond += '\nAND bayes1_prob IS NULL'

        # XXX For testing purpose
        # cond += '\nAND datetime >= 1509580799'

        c.execute('''SELECT datetime, freq_khz, char1_max_sn, char1_best_pos_hz,
                char1_total_ct, char1_bg_pos_hz
            FROM received
            %s
            ORDER BY datetime
            LIMIT %d''' % (cond, limit))

        n_rows = 0
        for row in c.fetchall():
            pprob = bayes(bi, row[0], row[1], row[2], row[3], row[4], row[5],
                debug=debug)
            n_rows += 1
            c.execute('''UPDATE received SET
                bayes1_prob = ?
                WHERE datetime = ?''',
                (
                    pprob,
                    row[0]
                ))
            conn.commit()

        if onepass and n_rows == 0:
            break
        else:
            # For continuous passes, 'force fetch' is NOT required
            force = False
            # To let rest database, wait for a short time period
            time.sleep(0.5)

    conn.close()

def task():
    """
    Entry point for Task Keeper
    """
    bayes_all(onepass=False, limit=1000, force=False, debug=False)

def main():
    import argparse
    import re
    import sys

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Bayesian Inference Program')
    parser.add_argument('-d', '--debug',
        action='store_true',
        default=False,
        help='enable debug')
    parser.add_argument('--force',
        action='store_true',
        default=False,
        help='update database even they already have inference')
    parser.add_argument('-q', '--quit',
        action='store_true',
        default=False,
        help='quit after one-pass')
    args = parser.parse_args()

    bayes_all(onepass=args.quit, limit=1000, force=args.force, debug=args.debug)

if __name__ == "__main__":
    main()
