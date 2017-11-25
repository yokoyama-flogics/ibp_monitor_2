"""
Graph Generator

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

import gd
import os
import sys

# Set Python search path to the parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.common import eprint

lborder = 9
rborder = 31
tborder = 18
bborder = 8 
cwidth = 5
cheight = 5
sskip = 17

def hsv_to_rgb(h, s, v):
        import math
        hi = int(math.floor(h / 60.0)) % 6
        f = h / 60.0 - hi
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)
        p = int(math.floor(p * 255.0 + 0.5))
        q = int(math.floor(q * 255.0 + 0.5))
        t = int(math.floor(t * 255.0 + 0.5))
        v = int(math.floor(v * 255.0 + 0.5))
        if hi == 0:
                r = v
                g = t
                b = p
        elif hi == 1:
                r = q
                g = v
                b = p
        elif hi == 2:
                r = p
                g = v
                b = t
        elif hi == 3:
                r = p
                g = q
                b = v
        elif hi == 4:
                r = t
                g = p
                b = v
        elif hi == 5:
                r = v
                g = p
                b = q
        else:
                r = 0
                g = 0
                b = 0
        # print r, g, b
        return (r, g, b)

def getindex(sn, pp, bias):
        sni = int(sn / 3)
        if sni < 0:
                sni = 0
        if sni > 5:
                sni = 5

        ppi = int((pp - 0.5) / 0.1)
        if ppi < 0:
                ppi = 0
        if ppi > 4:
                ppi = 4

        if bias < -5.0:
                bii = 0
        elif bias < -3.0:
                bii = 1
        elif bias < -1.0:
                bii = 2
        elif bias < 1.0:
                bii = 3
        elif bias < 3.0:
                bii = 4
        elif bias < 5.0:
                bii = 5
        else:
                bii = 6

        return (sni, ppi, bii)

def iminit(im, colidx):
        import os
        global nosig
        global white
        global black
        global darkgray
        callsigns = (
                '4U1UN (United Nations)',
                'VE8AT (Canada)',
                'W6WX (United States)',
                'KH6RS (Hawaii)',
                'ZL6B (New Zealand)',
                'VK6RBP (Australia)',
                'JA2IGY (Japan)',
                'RR9O (Russia)',
                'VR2B (Hong Kong)',
                '4S7B (Sri Lanka)',
                'ZS6DN (South Africa)',
                '5Z4B (Kenya)',
                '4X6TU (Israel)',
                'OH2B (Finland)',
                'CS3B (Madeira)',
                'LU4AA (Argentina)',
                'OA4B (Peru)',
                'YV5B (Venezuela)')

        for sni in range(6):
                for ppi in range(5):
                        for bii in range(7):
                                colidx[(sni, ppi, bii)] = \
                                        im.colorAllocate(hsv_to_rgb(
                                                120 + (bii - 3) * 20,
                                                ppi * 0.2 + 0.2,
                                                sni * 0.1 + 0.5))

        nosig          = im.colorAllocate(hsv_to_rgb(120, 0.0, 0.5))
        white          = im.colorAllocate(hsv_to_rgb(  0, 0.0, 1.0))

        bg             = im.colorAllocate(hsv_to_rgb(  0, 0.0, 0.7))
        black = im.colorAllocate((0, 0, 0))
        darkgray = im.colorAllocate(hsv_to_rgb(0, 0, 0.2))
        # im.colorTransparent(-1)
        im.fill((0, 0), bg)
        for i in range(18):
                x1 = lborder - 1
                y1 = tborder - 1 + i * (cheight * 5 + sskip)
                x2 = lborder + cwidth * 96 - 1
                y2 = tborder + i * (cheight * 5 + sskip) + cheight * 5 - 1
                im.rectangle((x1, y1), (x2, y2), darkgray)
                im.string(gd.gdFontMediumBold, (x1, y1 - 13), callsigns[i], black)
                for j in range(12, 95, 12):
                        im.line((x1 + cwidth * j, y1), (x1 + cwidth * j, y2), \
                                darkgray)
                im.string(gd.gdFontTiny, (x2 + 6, y1 - 1), "10m", black)
                im.string(gd.gdFontTiny, (x2 + 6, y1 - 1 + cheight * 4), "20m", black)
                        

        # for i in range(0, 50, 5):
        #       im.filledRectangle((0, 50 - i), (5, 50 - i + 5), colidx[i])

def imputmark(im, tindex, bindex, sindex, colidx):
        import math
        global nosig
        global white

        x = lborder + tindex * cwidth
        y = tborder + sindex * (cheight * 5 + sskip) + bindex * cheight
        # print x, y, col, dB
        # print (x, y), (x + 8, y + 8), colidx[col]
        if tindex % 4 == 3:
                split = 2
        else:
                split = 0
        im.filledRectangle((x, y), (x + cwidth - 2, y + cheight - 2), colidx)

def gen_graph(datestr, outfile_name, format=None, debug=False):
    """
    Generate graph
    outfile_name can be StringIO.  In the case, format must be 'PNG' or 'GIF'
    if outfile_name is string (or file name), format will be ignored
    """
    from datetime import datetime, timedelta
    from lib.fileio import connect_database
    from lib.ibp import freq_khz_to_mhz, get_slot
    import re

    im = gd.image((lborder + rborder + cwidth * 96 - 1, \
        tborder + bborder + 18 * (cheight * 5 + sskip) - sskip))
    colidx = {}
    iminit(im, colidx)

    conn = connect_database()
    c = conn.cursor()

    def datetime_to_sec(t):
        return int((t - datetime.utcfromtimestamp(0)).total_seconds())

    timefrom = datetime.strptime(datestr, '%Y%m%d')
    timeto   = timefrom + timedelta(days=1) - timedelta(seconds=10)

    if debug:
        print timefrom, timeto
        print datetime_to_sec(timefrom), datetime_to_sec(timeto)

    c.execute('''SELECT datetime, freq_khz, char1_max_sn, char1_best_pos_hz,
            bayes1_prob
        FROM received
        WHERE datetime >= ? AND datetime <= ? AND bayes1_prob IS NOT NULL''',
        (datetime_to_sec(timefrom), datetime_to_sec(timeto)))

    for row in c.fetchall():
        if debug:
            print row

        tindex = (row[0] % (3600 * 24)) / (15 * 60)
        bindex = {
            14100: 4,
            18110: 3,
            21150: 2,
            24930: 1,
            28200: 0
        }[row[1]]

        band = freq_khz_to_mhz(row[1])
        sindex = get_slot(row[0], band)
        sn = row[2]
        bias = float(row[3]) / band
        pp = row[4]

        found = (pp >= 0.5)
        if found:
                # print "found", pp
                imputmark(im, tindex, bindex, sindex, colidx[
                        getindex(sn, pp, bias)])
        else:
                imputmark(im, tindex, bindex, sindex, nosig)

    if type(outfile_name) is not str:
        import StringIO

        if format == 'PNG':
            writer = im.writePng
        elif format == 'GIF':
            writer = im.writeGif
        else:
            raise Exception('Unknown output file format')

        fimg = outfile_name
        writer(fimg)
        fimg.seek(0)

    else:
        if re.search('\.png$', outfile_name, flags=re.IGNORECASE):
            writer = im.writePng
        elif re.search('\.gif$', outfile_name, flags=re.IGNORECASE):
            writer = im.writeGif
        else:
            raise Exception('Unknown output file format')
        fimg = open(outfile_name, "wb")
        writer(fimg)
        fimg.close()

def task():
    """
    Continuously generate PNG graph files
    """

def main():
    import argparse
    import re
    import sys

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Beacon Reception Graph Generator')
    parser.add_argument('-d', '--debug',
        action='store_true',
        default=False,
        help='enable debug')
    parser.add_argument('-o', '--output',
        required=True,
        help='output file name (PNG graphic)')
    parser.add_argument('datestr',
        help='datestr (e.g. 20171028)')
    args = parser.parse_args()

    m = re.match(r'[0-9]{8}$', args.datestr)
    if not m:
        eprint("Illegal datestr '%s' specified" % (args.datestr))
        sys.exit(1)

    gen_graph(datestr=args.datestr, outfile_name=args.output, debug=args.debug)

if __name__ == "__main__":
    main()
