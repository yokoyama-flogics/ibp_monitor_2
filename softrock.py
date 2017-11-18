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

def get_version():
    context = usb1.USBContext()
    handle = context.openByVendorIDAndProductID(
        VENDOR_ID, PRODUCT_ID, skip_on_error=True)

    version = handle.controlRead(
        usb1.TYPE_VENDOR | usb1.RECIPIENT_DEVICE | usb1.ENDPOINT_IN,
        0,
        0xe00,
        0,
        2)

    # Notice that bytes are swapped
    major = version[1]
    minor = version[0]
    return major, minor

def softrock(debug=False):
    if debug:
        ver = get_version()
        print 'Attached SoftRock firmware is version %d.%d' % ver

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
