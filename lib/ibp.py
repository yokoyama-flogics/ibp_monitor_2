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

class Station:
    """
    Represents IBP beacon stations
    """
    def __init__(self):
        from lib.config import BeaconConfigParser
        import yaml
        yamlfile = BeaconConfigParser().get('Common', 'stations')
        self.stations = yaml.load(open(yamlfile, 'r'))

def main():
    pass

if __name__ == "__main__":
    main()
