"""
Beacon Monitor Task Keeper
"""

import logging
import os
import sys

# Set Python search path to the parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.common import eprint

def keep_task(func, name, sleep_sec=1):
    """
    Iterate function func and restart when it exited or raised an exception
    """
    from time import sleep

    def datestr():
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    while True:
        try:
            func()
            eprint('Function %s exited at %s.  Continued.' % (name, datestr()))

        except KeyboardInterrupt:
            break

        except:
            eprint('Function %s raised an exception at %s.  Continued.' % \
                (name, datestr()))
            logging.exception(name + ' at ' + datestr())

        sleep(sleep_sec)

def task_keeper(debug=False):
    """
    Run all tasks listed in the config file and monitor them
    """
    from lib.config import BeaconConfigParser
    from multiprocessing import Process
    from time import sleep

    logging.basicConfig(
        filename=BeaconConfigParser().getpath('TaskKeeper', 'logfile'))

    task_names = BeaconConfigParser().get('TaskKeeper', 'tasks').split(',')

    proc = {}
    for task in task_names:
        exec 'from ' + task + ' import task as f'
        proc[task] = Process(
            target=keep_task,
            args=(eval('f'), task + '.task()'))
        proc[task].start()

    try:
        while True:
            sleep(1)

    except KeyboardInterrupt:
        eprint('Interrupted by user.  Aborted.')

    for task in task_names:
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
