# How to install Beacon Monitor-2 to Raspbian

## Required gadgets

- Raspberry Pi 3 + appropriate micro SD card (4GB is enough if you use
  USB memory stick for database storage, but 16GB is typical because
  it isn't more expensive than smaller ones nowadays)

- SoftRock Receiver (SoftRock RX Ensemble II HF Receiver Kit of Five
  Dash Inc. may work... I tested it only by ancient 'RX Lite + USB
  Xtall V9.0 + Programmable BPFs', and also comparatively new 'RX
  Ensemble III HF Receiver'.)

- Behringer UCA202 (or other TI PCM2902 chip equipped USB ADC
  interface), and audio cables to connect to SoftRock

- Internet connection (Beacon Monitor requires precise time
  synchronization by Network Time Protocol).

- and some antenna. :)

## Installation

1. Install Raspbian Stretch Lite (2017-09-07 version) as explained at
   https://www.raspberrypi.org/documentation/installation/installing-images/ .

2. [option] If you don't like that Raspbian automatically extend file
   system in the SD card, refer
   https://www.raspberrypi.org/forums/viewtopic.php?p=977050, and
   remove 'quiet init=/usr/lib/raspi-config/init_resize.sh' in the
   boot/cmdline.txt of the SD card.

3. [recommended] If you don't want connect USB keyboard and HDMI
   monitor to Raspberry Pi, and also you have a USB serial adapter for
   Raspberry Pi to login, you need to add the line 'enable_uart=1' at
   the bottom of boon/config.txt .

4. Insert the SD card to Raspberry Pi, connect Ethernet cable, and
   power the board.

5. Log in by user 'pi' and password 'raspberry', and find assigned IP
   address to the Raspberry Pi.  Use command 'ifconfig' and find a
   line like

   ````
   inet 192.168.1.75  netmask 255.255.255.0  broadcast 192.168.1.255
   ````

   The '192.168.1.75' is assigned IP address of the Raspberry Pi.
   '127.0.0.1' is a local address and not the case.  Hereafter, we
   assume our Raspberry Pi has this IP address in this document.

6. [recommended] You can enable SSH server to login via Ethernet, run
   'sudo raspbi-config', choose options 'Interfacing Options' ->
   'SSH', and enable it.  PLEASE TAKE CARE that your Raspberry Pi may
   be controlled by intruders from the Internet.  You should change
   password at least.  (Firewall should be placed between the Internet
   and your Raspberry Pi.)

7. Hereafter, you can login by SSH if you enabled SSH server, like
   'ssh pi@192.168.1.75' from your PC.  Please find, by Google, where
   you can install an SSH client if you use Windows.  SSH client is
   already installed if you use Mac OS or Linux.

8. Connect a Behringer UCA202 to Raspberry Pi.  If you chose a
   different model, please find a solution by the Internet.  Google
   would help.  TI PCM2902 chip equipped model is strongly recommended
   because it was tested by Beacon Monitor 1 and 2, for long time.

10. Now, it is the time to install various tools which Beacon
   Monitor-2 requires.  Follow the steps below.

    The tool 'screen' is quite helpful to operate Beacon Monitor.  You
    can logout Raspberry Pi while running Monitor while monitoring the
    status messages.

    If you see any errors, it is recommended to run 'sudo apt-get
    update' once again, and retry the steps.

    ````$ sudo apt-get update````
	
    ````$ sudo apt-get install -y python-scipy````
	
    ````$ sudo apt-get install -y git````

    ````$ sudo apt-get install -y screen````
	
    ````$ sudo apt-get install -y libasound2-dev````
	
    ````$ sudo apt-get install -y python-gd````
	
    ````$ sudo apt-get install -y python-pip````
	
    ````$ sudo pip install libusb1````
	
    ````$ sudo pip install flask````
	
    ````$ sudo pip install pyyaml````

11. Next, we need the latest pyalsaaudio.  It required a slightly
    complicated procedure.
	
    ````$ mkdir ~/tmp && cd ~/tmp````

    ````$ wget https://pypi.python.org/packages/52/b6/44871791929d9d7e11325af0b7be711388dfeeab17147988f044a41a6d83/pyalsaaudio-0.8.4.tar.gz```` (Please exactly type as a long command without line-breaking.)

    ````$ tar xzf pyalsaaudio-0.8.4.tar.gz````

    ````$ cd pyalsaaudio-0.8.4````

    ````$ sudo python setup.py install````

12. Finally obtain Beacon Monitor-2 code from GitHub.

    ````$ cd && git clone https://github.com/yokoyama-flogics/ibp_monitor_2.git````

    ````$ cd ibp_monitor_2````

    ````$ screen````

    (Press space or return key.)

11. We need permission to access USB device (or SoftRock), so execute
    the following command.  (Please exactly type as a long command
    without line-breaking.)
	
	````$ sudo sh -c 'echo ATTRS{idVendor}==\"16c0\", ATTRS{idProduct}==\"05dc\", SUBSYSTEMS==\"usb\", ACTION==\"add\", MODE=\"660\", GROUP=\"users\" > /etc/udev/rules.d/99-softrock.rules'````
	
	After that, you will see the following result:
	
	````$ cat /etc/udev/rules.d/99-softrock.rules````

    ````ATTRS{idVendor}=="16c0", ATTRS{idProduct}=="05dc", SUBSYSTEMS=="usb", ACTION=="add", MODE="660", GROUP="users"````

9. Connect SoftRock to Raspberry Pi.  My personal recommended steps
   is, first apply power by DC jack (should be 12V.  Please consult
   with SoftRock documentation), and connect SoftRock by USB cable
   later.

11. Next, you need to choose which version of Beacon Monitor-2 you
    want to run.  At the moment, let's use the branch
    ````beta```` because that is the only release at the
    moment.  (Sorry.)  Checkout the source code.
	
    ````$ git checkout beta````

12. Now, you should configure how Beacon Monitor-2 operates.
    Configurations are described in the file 'bm2.cfg'.  Please open
    the file by your favorite text editor.  If you aren't familiar
    with Linux, editor 'nano' is recommended.
	
	````$ nano bm2.cfg````
	
	In the section 'Common', the keyword 'database' specifies where
    the database file is located.  Default should be okay, but you may
    consider to place it in the external USB memory stick because the
    database file 'bm2.db' is frequently read and written, so the SD
    card wears very quickly.  In any cases, you should backup the
    bm2.db periodically not to loose the valuable past records in the
    database.
	
	In the section 'Signal', the keyword 'dir' tells where recorded
    signal (wav files) are stored.  The signal files will not be
    referred anymore, once Bayesian Inference of signal files are
    completed.  However, it may be helpful for some reason, for
    example, you want rerun Bayesian Inference, or want to
    signal-process the file and convert them to hearable sound files.
    (Not implemented yet.)  The files exhaust SD card (or USB memory)
    very quickly, you may want to consider place them into the
    external USB memory or RAM disk.
	
	In the section 'SignalRecorder', you see the keyword
    'calib_softrock'.  This tells Si570 (clock generator of SoftRock)
    frequency bias which is different from device to device.  You can
    leave the value to 0.0, but if you have accurate receiver, you can
    specify an appropriate value.  It is a good idea to use WWV or
    similar station which is equipped with atomic clock.
	
	The keyword 'lrlag' in the same section is important.  This is
    required because TI PCM2902 chip has an erratum that R channel
    output (of ADC) is always one sample behind L channel.  The
    specified number compensates the delay.  If your sound card
    doesn't equip PCM2902 and the device doesn't have a similar
    erratum, the value must be zero (0).
	
	The keyword 'sig_iq' is also important.  This specifies which
    channel (L or R) corresponds to 'I' or 'Q' of quadrature
    demodulation each other.  Some people say SoftRock output is 'L =
    Q and R = I', and I agree with it, so this should be 'R/L'.  In
    the case, you didn't detect any beacons, I/Q connections may be
    wrong.  In the case, please try 'L/R' instead.
	
	In the section 'TaskKeeper', the keyword 'logfile' specifies where
    Task Keeper outputs log.  The keyboard 'timelimit_sec' means how
    long (in seconds) signal wav files are preserved.  Regardless the
    value, files which are not processed by Bayesian Inference won't be
    removed.
	
	In the section 'GraphServ', the 'port' specified the web server
    HTTP port.  If the value is 5000, you can access the generated
    graph like 'http://192.168.1.75:5000/graph/20171121.png'.  It is
    explained again later.
	
13. Next, we need to initialize database.  It is required just ONCE.
    If you repeat this, database will be CLEARED!  Please take care.

    ````$ python initdb.py agree````

    (Answer yes.)

14. Finally run Signal Recorder.  It talks to SoftRock and USB sound
    adapter, and receive signal and store that as signal (.wav) files
    and also to database.
	
	````$ python sigrec_softrock.py -d````
	
	At the moment, the option '-d' (means debug enabled) is
    recommended.
	
15. Please be patient for a minute.  You will see diagnostics like below.

    ````
    datetime_sec_to_freq_khz: minute=57
    Changing frequency: 28200 kHz
    datetime_sec_to_freq_khz: minute=57
    Changing frequency: 28200 kHz
    /home/pi/signal/20171123/035730.wav
    160000 (160000, 2)
    register_db: 1511409450
    ````

    Signal Recorder is working if you see them.  The are repeated for
    every 10 seconds.

16. Next, we need to run various tasks.  In brief,

    - Characteristics Extractor (charex.py)
    - Bayesian Inference (bayes.py)
    - Signal Files Clearner (cleaner.py)
    - Graph Image Web Server (graphserv.py)
	
    You can run them easily by Task Keeper.

    However, at the moment, Signal Recorder is running as a foreground
    command.  So the command 'screen' is required.  (It will be
    improved in the near future so that you can run Signal Recorder as
    a background process.)  Anyway, in the meanwhile...

    Type a key sequence Ctrl-a and Ctrl-c.  The 'screen' command
    launches another screen and a command shell.  The screen becomes
    blank and you will see a new command prompt.  Don't panic.  You
    can switch back to the previous screen by typing Ctrl-a and
    Ctrl-n.  By repeating Ctrl-a and Ctrl-n, you can toggle the
    screens.
	  
    On the new screen and command prompt, we run Task Keeper.

    ````$ python taskkeeper.py -d````

    You will see a spinning cursor below the command line.

17. Check if tasks are working.

    Now we can see if beacon monitoring result graphs can be obtained.
    You run any your favorite web browser on PC.  Then access the following URL.

    - http://192.168.1.75:5000/graph/today.png

    or

    - http://192.168.1.75:5000/graph/20171123.png

    or

    - http://192.168.1.75:5000/graph/20171123.gif

    The first example gets the today's result by PNG format.  Second
    one is specifying an exact date.  The last one is specifying GIF
    format instead of PNG.

    IMPORTANT NOTICE: You should NOT export the beacon monitor web
    server's port publicly.  Flask (the HTTP server component) is not
    recommending to do so.  Please run another robust web server
    (Apache etc.) and put Beacon Monitor web server behind it.

18. Now you can logout Raspberry Pi.  Once again, the command 'screen'
    is great to realize logging-out from Raspberry Pi without
    terminating Beacon Monitor tasks.
	
	You simply type Ctrl-a and Ctrl-d.  You see a message like
    '[detached from ...]'.  Now you can logoff by typing 'exit'.
	
	When you want to login and monitor commands (Signal Recorder and
    Task Keeper), you login Raspberry Pi and execute command 'screen
    -r'.  You can take the sessions back.  Key sequence Ctrl-a and
    Ctrl-n should also work.
	
## If you have any questions

Please contact me by email.
Email address is shown in the 'Links' list on the site
http://ayoko.net/beacon/ .
