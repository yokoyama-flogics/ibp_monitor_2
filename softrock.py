"""
SoftRock Controller

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

VENDOR_ID = 0x16c0
PRODUCT_ID = 0x05dc
I2C_ADDR = 0x55         # Mandatory for old version firmware
TIMEOUT = 500

import usb1
import os
import sys

# Set Python search path to the parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.common import eprint

def softrock_handle():
    """
    Return USB handler talking to SoftRock
    """
    context = usb1.USBContext()
    return context.openByVendorIDAndProductID(
        VENDOR_ID, PRODUCT_ID, skip_on_error=True)

def get_version():
    # Check if version information is already cached
    if not hasattr(get_version, 'version'):
        try:
            # Refer https://pe0fko.nl/SR-V9-Si570/
            get_version.version = softrock_handle().controlRead(
                usb1.TYPE_VENDOR | usb1.RECIPIENT_DEVICE | usb1.ENDPOINT_IN,
                0,
                0xe00,
                0,
                2,
                TIMEOUT)
        except:
            eprint('ERROR: Confirm SoftRock is attached to a USB port and also'
                ' you have a privilege\n       to access the USB port.')
            sys.exit(1)

    # Notice that bytes are swapped
    major = get_version.version[1]
    minor = get_version.version[0]
    return major, minor

def initialize(debug=False):
    """
    If SoftRock firmware is 14.0, some historical procedure is carried out.
    Because specification is unknown, I don't know (or forgot) if this is
    still required.  Served for historical reason.
    """
    if get_version()[0] >= 15:
        return

    if debug:
        print "Initializing SoftRock because this formware version is too old."

    handle = softrock_handle()

    handle.controlRead(
        usb1.TYPE_VENDOR | usb1.RECIPIENT_DEVICE | usb1.ENDPOINT_IN,
        0,
        0x1234,
        0x5678,
        8,
        1000)

    handle.controlRead(
        usb1.TYPE_VENDOR | usb1.RECIPIENT_DEVICE | usb1.ENDPOINT_IN,
        1,
        0x30,
        0,
        8,
        1000)

def set_freq(freq_hz, dryrun=False, debug=False):
    """
    Set SoftRock to the specified freq_hz
    If attached SoftRock has old firmware, we need special cares.
    """
    from lib.config import BeaconConfigParser
    import math
    bytes = [0, 0, 0, 0]

    if debug:
        print 'freq_hz = %d' % (freq_hz)

    if get_version()[0] < 15:
        # This came from the Monitor-1 code.  I don't know why this calculation
        # is required because firmware 14.0 documentation is missing.
        CALIB = 2.0962539700083447          # 2013-09-16 27.9 deg
        v = float(freq_hz)
        v *= CALIB
        v *= 4                              # Firmware 14.0 requires this
        ival = int(math.floor(v + 0.5))

    else:
        calib = eval(
            BeaconConfigParser().get('SignalRecorder', 'calib_softrock'))
        if debug:
            print 'calib = %g' % (calib)

        # Referred http://www.n8mdp.com/sdrLinux.php and usbsoftrock source code
        # on the site
        MULT = 4
        freq_hz *= (1.0 + calib) / 1e6 * (1 << 21) * MULT
        ival = int(freq_hz + 0.5)

    if debug:
        print 'ival = %d' % (ival)

    bytes[0] = ival         & 0xff
    bytes[1] = (ival >> 8)  & 0xff
    bytes[2] = (ival >> 16) & 0xff
    bytes[3] = (ival >> 24) & 0xff

    if debug:
        print 'bytes =', bytes

    if not dryrun:
        softrock_handle().controlWrite(
            usb1.TYPE_VENDOR | usb1.RECIPIENT_DEVICE | usb1.ENDPOINT_OUT,
            0x32,
            0x700 + I2C_ADDR,
            0,
            bytes,
            TIMEOUT)

    if get_version()[0] < 15:
        # This version may not have BPF automatic setting.
        if debug:
            print 'This SoftRock may not have BPF automatic setting.'
            print 'Using a manual way.'

        if freq_hz < 4000000:
            bpf = 0
        elif freq_hz < 8000000:
            bpf = 1
        elif freq_hz < 16000000:
            bpf = 2
        else:
            bpf = 3

        if debug:
            print 'bpf = %d' % (bpf)

        # This came from the Monitor-1 code.
        bytes = [0] * 8
        if not dryrun:
            softrock_handle().controlWrite(
                usb1.TYPE_VENDOR | usb1.RECIPIENT_DEVICE | usb1.ENDPOINT_IN,
                4,
                (bpf & 0x3) << 4,
                0,
                bytes,
                1000)

def main():
    import argparse
    import re
    import sys

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='SoftRock Control Program')
    parser.add_argument('-d', '--debug',
        action='store_true',
        default=False,
        help='enable debug')
    parser.add_argument('--dryrun',
        action='store_true',
        default=False,
        help='dry-run')
    parser.add_argument('freq',
        help='frequency in Hz')
    args = parser.parse_args()

    if args.debug:
        print 'Attached SoftRock firmware is version %d.%d' % get_version()

    initialize(debug=args.debug)
    set_freq(int(args.freq), dryrun=args.dryrun, debug=args.debug)

if __name__ == "__main__":
    main()
