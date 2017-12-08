# Frequently Asked Questions

1. Where are the daily graphs stored?

   **Answer**: Different from Faros, Beacon Monitor-2 doesn't
   periodically make daily graphs.  Instead, we need to use
   `graphserv.py` (graph generating web server), or `gengraph.py`
   manually.
   
   In the former case, when we access the web server port (5000 by
   default), a daily graph is generated and shown to the web client
   (or web browser).

   In the latter case, you can manually run the following command to
   get daily graph of 2017-12-07.
   
   ````
   $ python gengraph.py -o foobar.png 20171207
   ````
   
   The foobar.png can be an arbitrary name.  If you specify extension
   '.gif', GIF file is generated instead.

2. Does the audio (I/Q) input level make any difference to the
   sensitivity and selectivity of the Rx?
   
   **Answer**: Beacon Monitor-2 software mainly makes decision by
   relative signal level difference (or S/N, Signal to Noise ratio),
   so it shouldn't affect the results basically.  Theoretically, it is
   okay if strong signal doesn't saturate A/D converter and also the
   background noise can be detected by A/D converter.
   
   It may be helpful to look into the database regarding signal
   levels.  To do so, SQLite3 command tool is required.  If command
   'sqlite3' is installed, you can try
   
   ````
   $ sqlite3 bm2.db .schema
   CREATE TABLE `received` (
        `datetime`              INTEGER UNIQUE,
        `offset_ms`             INTEGER,
        `freq_khz`              INTEGER,
        `bfo_offset_hz`         INTEGER,
        `recorder`              TEXT,
        `char1_max_sn`          REAL,
        `char1_best_pos_hz`     INTEGER,
        `char1_total_ct`        INTEGER,
        `char1_bg_pos_hz`       INTEGER,
        `char1_bg_sn`           REAL,
        `bayes1_prob`           REAL,
        PRIMARY KEY(`datetime`)
   );
   ````
   
   If you see error 'database is locked', you can repeat the command.
   The above command show record format of the database.
   
   Next, you try
   
   ````
   $ sqlite3 bm2.db .dump | head -100
   ````
   
   'head -100' retrieves the very first records in the database.  If
   you change it to 'tail -100', you will see the latest records in
   the database.
   
   In anyway, the above command dumps following information or
   similar.
   
   ````
   INSERT INTO "received" VALUES(1511164850,-1000,14100,3650,'SRockV9Lite+PCM2902',1.79362686437832419272e+00,48,1,-157,-6.06484112449413137468e-01,2.84541914624008825465e-03);
   ````
   
   The value 1.79362686437832419272e+00 corresponds to `char1_max_sn`
   and the value -6.06484112449413137468e-01 corresponds to
   `char1_bg_sn`.  In other words, the record shows S/N = 1.8 dB and
   the background-noise time period (pseudo) S/N = -0.61 dB.
   Especially, the latter is important.  When you compare the
   `char1_bg_sn` of multiple records in the database, and the values
   don't vary so much, the receiving signal level may be too small.
   
   Another idea is, reading signal files (.wav) by some tools like
   Audacity.  Even you sufficiently magnify (or amplify) the signal,
   if you don't see any noises, you may need some amplifier between
   SDR receiver (or SoftRock) and ADC.
   
   However, to my experience, output level of SoftRock looks high
   enough for my ADC (Behringer UCA202).
