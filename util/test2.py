import time

def task():
    ct = 0
    while True:
        print 'test2 task...'
        ct += 1
        if ct > 5:
            break

        time.sleep(2)
