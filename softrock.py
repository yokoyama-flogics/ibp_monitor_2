"""
SoftRock Controller
"""

VENDOR_ID = 0x16c0
PRODUCT_ID = 0x05dc

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
    # Refer https://pe0fko.nl/SR-V9-Si570/
    version = softrock_handle().controlRead(
        usb1.TYPE_VENDOR | usb1.RECIPIENT_DEVICE | usb1.ENDPOINT_IN,
        0,
        0xe00,
        0,
        2)

    # Notice that bytes are swapped
    major = version[1]
    minor = version[0]
    return major, minor

def set_freq(freq_hz, debug=False):
    """
    Set SoftRock to the specified freq_hz
    If attached SoftRock has old firmware, we need special cares.
    """
    if get_version()[0] < 15:
        # This version may not have BPF automatic setting.
        if debug:
            print 'This SoftRock may not have BPF automatic setting.'
            print 'Using a manual way.'

def softrock(debug=False):
    try:
        ver = get_version()
    except:
        eprint('ERROR: Confirm SoftRock is attached to a USB port and also you'
            ' have a privilege\n       to access the USB port.')
        sys.exit(1)

    if debug:
        print 'Attached SoftRock firmware is version %d.%d' % ver

    set_freq(100000, debug)

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
    args = parser.parse_args()

    softrock(debug=args.debug)

if __name__ == "__main__":
    main()
