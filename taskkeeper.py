"""
Beacon Monitor Task Keeper

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

import logging
import os
import sys

# Set Python search path to the parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.common import eprint

def keep_task(func, name, exit_return=False, sleep_sec=1):
    """
    Iterate function func and restart when it exited or raised an exception
    if the function completed, it will be restarted if exit_return is True
    """
    from time import sleep

    def datestr():
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    while True:
        try:
            func()
            if exit_return:
                return

            eprint('Function %s exited at %s.  Continued.' % (name, datestr()))
            logging.error('Function %s exited at %s.  Continued.' % \
                (name, datestr()))

        except KeyboardInterrupt:
            break

        except:
            eprint('Function %s raised an exception at %s.  Continued.' % \
                (name, datestr()))
            logging.exception(name + ' at ' + datestr())

        sleep(sleep_sec)

def spinning_cursor():
    while True:
        for cursor in '|/-\\':
            yield cursor

def task_keeper(debug=False):
    """
    Run all tasks listed in the config file and monitor them
    """
    from lib.config import BeaconConfigParser
    from multiprocessing import Process
    from time import sleep
    import sys

    config = BeaconConfigParser()

    logging.basicConfig(filename=config.getpath('TaskKeeper', 'logfile'))
    task_names = config.get('TaskKeeper', 'tasks').split(',')

    proc = {}
    for task in task_names:
        if task[0] == '@':
            exit_return = True
            task = task[1:]
        else:
            exit_return = False

        exec 'from ' + task + ' import task as f'
        proc[task] = Process(
            target=keep_task,
            args=(eval('f'), task + '.task()', exit_return))
        proc[task].start()

    try:
        spinner = spinning_cursor()
        while True:
            sys.stdout.write(spinner.next())
            sys.stdout.write('\b')
            sys.stdout.flush()
            sleep(0.25)

    except KeyboardInterrupt:
        eprint('Interrupted by user.  Aborted.')

    for task in task_names:
        if task[0] == '@':
            task = task[1:]

        proc[task].join()

def main():
    import argparse
    import re
    import sys

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Beacon Monitor Task Keeper')
    parser.add_argument('-d', '--debug',
        action='store_true',
        default=False,
        help='enable debug')
    args = parser.parse_args()

    task_keeper(debug=args.debug)

if __name__ == "__main__":
    main()
