# Common Troubleshooting

## Found errors when launched 'sigrec_softroch.py'

Two common cases are known.  One is caused by USB connection to
SoftRock, and another is audio connection to ADC (audio adapter or
card) by ALSA library.  Explained in detail below.

### SoftRock USB error

If you see

````
ERROR: Confirm SoftRock is attached to a USB port and also you have a privilege
to access the USB port.
`````

the software 'softrock.py' wasn't able to talk to the micro-controller
in SoftRock.  (The micro-controller has responsibilities to set
oscillator Si570 and also change BPF (band-pass filter) on the board.)

To investigate the issue, in the directory ````ibp_monitor_2````,
there is tool named 'softrock.py'.  This is used to talk to SoftRock by
USB and change receiving frequency.  You can execute command

````
$ python softrock.py -d 10000000
````

This asks SoftRock to set the oscillator Si570 in the SoftRock, to
receive signals around 10 MHz.

If this showed an error, there should be a problem around libusb.

1. If this is the case, please try

   ````
   $ lsusb
   ````

   This lists what USB devices are connected to your PC.  If this list
   contains like:

   ````
   Bus 001 Device 029: ID 16c0:05dc Van Ooijen Technische Informatica shared ID for use with libusb
   ````

   The vendor ID and device ID must match the IDs which are described
   in softrock.py.

2. Second, the defined rule in the directory ````/etc/udev/rules.d````
   might be inappropriate.  You see the following line in the file
   ````99-softrock.rules```` if you followed the documentation.

    ````
    ATTRS{idVendor}=="16c0", ATTRS{idProduct}=="05dc", SUBSYSTEMS=="usb", ACTION=="add", MODE="660", GROUP="users"
    ````

    This should be okay for Raspbian (Raspberry Pi OS).  However, for
    other Linux distribution, the specified GROUP (at the last of the
    line) might need to be changed.

    Please try command 'id' as shown below.  This is my case on Ubuntu
    16.04.3.

    ````
    yokoyama@ubuntu:~$ id
    uid=1000(yokoyama) gid=1000(yokoyama) groups=1000(yokoyama),4(adm),24(cdrom),27(sudo),30(dip),46(plugdev),113(lpadmin),128(sambashare)
    ````

    My account belongs to group (gid) 1000 (or yokoyama), but NOT
    'users'.  So,

    ````
    ATTRS{idVendor}=="16c0", ATTRS{idProduct}=="05dc", SUBSYSTEMS=="usb", ACTION=="add", MODE="660", GROUP="users"
    ````

    doesn't work on (at least my) Ubuntu.

    The udev rule must be rewritten to

    ````
    ATTRS{idVendor}=="16c0", ATTRS{idProduct}=="05dc", SUBSYSTEMS=="usb", ACTION=="add", MODE="660", GROUP="yokoyama"
    ````

    In short, the last descriptor 'GROUP' needs to match your group
    ID.  Once again, your group ID is shown in the output of command
    'id', and the description 'gid=xxxx' corresponds to your group ID.
    If gid shows '1000(yokoyama)' (in my case), the udev rule must be
    as above, or

    ````
    ATTRS{idVendor}=="16c0", ATTRS{idProduct}=="05dc", SUBSYSTEMS=="usb", ACTION=="add", MODE="660", GROUP="1000"
    ````

IMPORTANT: Please notice that you need to disconnect and reconnect
SoftRock when you changed the udev rule.

### ALSA error

If you see an error like

````
ALSA lib pcm_hw.c:1700:(_snd_pcm_hw_open) Invalid value for card
````

specified ALSA card or device specification
````alsa_dev=hw:CARD=CODEC,DEV=0```` in bm2.cfg may be wrong.

To get a diagnostic, we need a tool named 'arecord'.  In the case it
is not installed, if you use Debian or Ubuntu, you can installed it by

````
$ sudo apt-get install alsa-utils
````

And please try

````
$ arecord -l
````

When I tried it on my Raspberry Pi, I saw

````
**** List of CAPTURE Hardware Devices ****
card 1: CODEC [USB Audio CODEC], device 0: USB Audio [USB Audio]
  Subdevices: 0/1
  Subdevice #0: subdevice #0
````

In the case, the description in bm2.cfg should be

````
alsa_dev=hw:CARD=CODEC,DEV=0
````

because arecord said the card name is 'CODEC' and device number is 0.
To clarify the problem in depth, 'aplay -l' may also help.  By my
Raspberry Pi, it shows

````
**** List of PLAYBACK Hardware Devices ****
card 0: ALSA [bcm2835 ALSA], device 0: bcm2835 ALSA [bcm2835 ALSA]
  Subdevices: 8/8
  Subdevice #0: subdevice #0
  Subdevice #1: subdevice #1
  Subdevice #2: subdevice #2
  Subdevice #3: subdevice #3
  Subdevice #4: subdevice #4
  Subdevice #5: subdevice #5
  Subdevice #6: subdevice #6
  Subdevice #7: subdevice #7
card 0: ALSA [bcm2835 ALSA], device 1: bcm2835 ALSA [bcm2835 IEC958/HDMI]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
card 1: CODEC [USB Audio CODEC], device 0: USB Audio [USB Audio]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
 ````

It means that Raspberry Pi also has embedded sound capability and the
card name is 'ALSA' and devices are 0 and 1.  In addition, the command
detected the card 'CODEC' and device ID 0 too.  The latter is the USB
sound adapter which is attached to my Raspberry Pi.  It also implies
that the embedded 'ALSA' (bcm2835) device has playback functionality
but it doesn't have 'capturing' (or recording) capability.

By the fact above, in the Raspberry Pi case, we need an external audio
device (or USB audio adapter).

If your sound card connected to Linux box is different from the above,
you might need to change the 'alsa_dev' configuration in bm2.cfg.
