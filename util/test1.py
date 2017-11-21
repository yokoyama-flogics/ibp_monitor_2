import time

def task():
    ct = 0
    while True:
        print 'test1 task...'
        ct += 1
        if ct > 5:
            raise Exception

        time.sleep(1)
