# Beacon Monitor-2 Project

This is software to monitor 'NCDXF/IARU International Beacon Project'
stations by Raspberry Pi and SoftRock SDR receiver.  Please also refer
http://www.ncdxf.org/beacon/

After the statement on the web site (http://ayoko.net/beacon/) on
March, 2017, the progress was quite slow until this November.
Sorry for your patience so far.

## Installation

Please refer [installation document](./doc/installation.md).

Currently, this software is provided as a beta version.  It still
requires clean-up work.

## Web site

- My own monitor station: http://ayoko.net/beacon

- Facebook page: https://www.facebook.com/beaconmonitor2/

## Recent progress background

I had decided to postpone rewriting the current "Signal Recorder",
which was written by
C language, and started rewriting other (or succeeding) signal
processing parts by Python language so that the project software
becomes more portable (or be able to run on many PC or Raspberry Pi).

However, as of Nov. 18, the parts have been mostly completed, so I'm
now rewriting the Signal Recorder code by Python and ALSA library.

My main receiver for Beacon Monitor is SoftRock and PCM2902 USB CODEC
adapter, so I'm focusing on it.

The current code can be found in the branch ```dev_start_20171016```
(https://github.com/yokoyama-flogics/ibp_monitor_2/tree/dev_start_20171016).

# Missing functionality

Functionality which is not implemented yet though there were in the
previous Monitor-1 site.

- Map Visualizer (gnomonic projection mapping)

- Twitter bot

Thank you for your additional patience...

Regards,

Atsushi Yokoyama (JN1SDD)
