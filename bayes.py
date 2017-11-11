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
            print '@@@@@', val
            self.hist[bin] += val
        else:
            print '<<<<<', val
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

        print '#@#', passed_sec, int(float(bias) / comp_band * 28), sn, ct, time_weight, sameband_weight
        self.add_element(int(float(bias) / comp_band * 28),
            float(sn) / 10 * \
            math.pow(float(ct) / 7.0, 2) * \
            math.pow(time_weight, 2) * sameband_weight)

    def result(self):
        sum = 0.0
        weight = 0.0

        if self.hist == {}:
            print "LARGE DIST"
            return 0.0, 1.0

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

    from lib.ibp import Station, get_slot, freq_khz_to_mhz
    from lib.fileio import connect_database

    identify = Station().identify_station

    # Identify transmitting station by time and band
    timeslot_in_sched, effective_time_sec, station = \
        identify(datetime_sec, freq_khz)
    print '<<<', timeslot_in_sched, effective_time_sec, station

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
            identify(candidate_datetime, candidate_freq_khz)
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
        print '!!!', row, candidate_station
        passed_sec = datetime_sec - candidate_datetime
        sn = row[2]
        bias = row[3]
        ct = row[4]
        stat.add_params(passed_sec, freq_khz, candidate_freq_khz, sn, bias, ct)

    print stat.hist
    print stat.result()

    print "---------------"

def bayes(datetime_sec, freq_khz, debug=False):
    """
    Bayesian Inference
    """
    import sys

    print '#', datetime_sec, freq_khz
    biashist(datetime_sec, freq_khz)

    # sys.exit(0)
    return None

def bayes_all(onepass=False, limit=1000, force=False, debug=False):
    """
    Retrieve any record in the database, which doesn't have Bayesian Inference
    this bayes.py yet, and pass them to bayes()
    """
    from lib.fileio import connect_database
    import time

    conn = connect_database()
    while True:
        c = conn.cursor()

        # If specified 'force', even the record has characteristics parameters,
        # fetch any records for update.
        if force:
            cond = ''
        else:
            # XXX
            # cond = 'WHERE datetime >= 1462762419 AND bayes1_sn IS NULL'
            cond = 'WHERE bayes1_sn IS NULL'

        c.execute('''SELECT datetime, freq_khz
            FROM received
            %s
            ORDER BY datetime
            LIMIT %d''' % (cond, limit))

        n_rows = 0
        for row in c.fetchall():
            paramset = bayes(row[0], row[1], debug=debug)
            n_rows += 1
            # paramset.updatedb(conn, row[0])

        if onepass and n_rows == 0:
            break
        else:
            # For continuous passes, 'force fetch' is NOT required
            force = False
            # To let rest database, wait for a short time period
            time.sleep(0.5)

    conn.close()

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
    parser.add_argument('--daemon',
        # nargs=1,
        choices=['start', 'stop', 'restart'],
        help='run as daemon.  start, stop, or restart')
    args = parser.parse_args()

    bayes_all(onepass=args.quit, limit=1000, force=args.force, debug=args.debug)

if __name__ == "__main__":
    main()
