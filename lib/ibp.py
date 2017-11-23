"""
IBP Parameters
"""

def callsign_to_region(callsign):
    """
    Convert callsign to where the station is located
    """
    return {
        '4U1UN': 'United Nations',
        'VE8AT': 'Canada',
        'W6WX': 'United States',
        'KH6WO': 'Hawaii',
        'KH6RS': 'Hawaii',
        'ZL6B': 'New Zealand',
        'VK6RBP': 'Australia',
        'JA2IGY': 'Japan',
        'RR9O': 'Russia',
        'VR2B': 'Hong Kong',
        '4S7B': 'Sri Lanka',
        'ZS6DN': 'South Africa',
        '5Z4B': 'Kenya',
        '4X6TU': 'Israel',
        'OH2B': 'Finland',
        'CS3B': 'Madeira',
        'LU4AA': 'Argentina',
        'OA4B': 'Peru',
        'YV5B': 'Venezuela'
    }[callsign]

def callsign_to_slot(callsign):
    """
    Convert callsign to IBP schedule time slot (0 ... 17)
    This is required for migration reason.
    """
    return {
        '4U1UN':  0,
        'VE8AT':  1,
        'W6WX':   2,
        'KH6WO':  3,
        'KH6RS':  3,
        'ZL6B':   4,
        'VK6RBP': 5,
        'JA2IGY': 6,
        'RR9O':   7,
        'VR2B':   8,
        '4S7B':   9,
        'ZS6DN': 10,
        '5Z4B':  11,
        '4X6TU': 12,
        'OH2B':  13,
        'CS3B':  14,
        'LU4AA': 15,
        'OA4B':  16,
        'YV5B':  17
    }[callsign]

def mhz_to_freq_khz(mhz):
    """
    Convert MHz to exact frequency in kHz
    """
    return {
        14: 14100,
        18: 18110,
        21: 21150,
        24: 24930,
        28: 28200
    }[mhz]

def freq_khz_to_mhz(freq_khz):
    """
    Convert frequency in kHz to band (or MHz value)
    """
    return freq_khz / 1000

def obtain_list_from_ncdxf():
    """
    Obtain beacon stations list from NXDXF web site
    """
    URL = 'http://ncdxf.org/beacon/programs/Beacon.lst'

    import StringIO
    import urllib2

    # Read the contents of the list
    data = urllib2.urlopen(URL).read()

    def round(x):
        import math
        return math.floor(float(x) * 100 + 0.5) / 100

    stations = []
    for line in StringIO.StringIO(data).readlines():
        stuple = line.rstrip().split('|')

        station = {
            'effective': '1970-01-01',
            'callsign': stuple[3],
            'longitude': round(float(stuple[1]) / 180),
            'latitude': round(float(stuple[2]) / 180)}

        slot = [station]
        stations.append(slot)

    return stations

def stations_to_yaml(stations):
    """
    Convert stations list (output of obtain_list_from_ncdxf()) to YAML string
    """
    # import yaml
    # pprint.PrettyPrinter(indent=4, depth=10).pprint(stations)
    # print yaml.dump(stations, Dumper=yaml.Dumper, default_flow_style=True)
    # print yaml.dump(stations, Dumper=yaml.Dumper, default_flow_style=False)
    # print yaml.dump(locals(), Dumper=yaml.Dumper, default_flow_style=False)

    s = ''
    slot_id = 0

    for slot in stations:
        s += '- # slot %d\n' % (slot_id)
        for station in slot:
            s += "  - effective: '%s'\n" % (station['effective'])
            s += "    callsign:  '%s'\n" % (station['callsign'])
            s += "    region:    '%s'\n" % \
                                    (callsign_to_region(station['callsign']))
            s += "    latitude:  %+7.2f\n" % (station['latitude'])
            s += "    longitude: %+7.2f\n" % (station['longitude'])
        slot_id += 1

    return s

def get_slot(datetime_sec, band):
    """
    Return IBP schedule time slot (0 ... 17) from given datetime_sec (second
    from UNIX time epoch) and band (14, 18, ..., 28) MHz value
    """
    time_xmit = 10  # sec (transmitting time length)
    n_slots = 18    # number of slots
    period_sched = n_slots * time_xmit

    slot_offset = {
        14: 0,
        18: 1,
        21: 2,
        24: 3,
        28: 4
    }

    timeslot_in_sched = int(datetime_sec % period_sched) / time_xmit
    return (timeslot_in_sched - slot_offset[band]) % n_slots

def datetime_to_sec_from_epoch(t):
    from datetime import datetime
    return int((t - datetime.utcfromtimestamp(0)).total_seconds())

class Station:
    """
    Represents IBP beacon stations
    """
    # XXX  Should be changed 'class IBP'?
    def __init__(self):
        from lib.config import BeaconConfigParser
        import yaml
        yamlfile = BeaconConfigParser().get('Common', 'stations')
        self.stations = yaml.load(open(yamlfile, 'r'))
    def identify_station(self, datetime_sec, freq_khz):
        """
        Identify transmitting station by datetime_sec (seconds from UNIX epoch)
        and received frequency (in kHz).
        """
        from datetime import datetime

        latest_effective_sec = -1
        latest_candidate = None

        slot = get_slot(datetime_sec, freq_khz_to_mhz(freq_khz))
        # debug  slot = 3 # get_slot(datetime_sec, band) # XXX

        for candidate in self.stations[slot]:
            candidate_effective_sec = \
                datetime_to_sec_from_epoch(
                    datetime.strptime(candidate['effective'], '%Y-%m-%d'))
            # print candidate, candidate_effective_sec
            if candidate_effective_sec < datetime_sec and \
                    candidate_effective_sec > latest_effective_sec:
                latest_effective_sec = candidate_effective_sec
                latest_candidate = candidate
                # print "found", latest_effective_sec
        # print "Result:", latest_candidate

        return slot, latest_effective_sec, latest_candidate

def main():
    pass

if __name__ == "__main__":
    main()
