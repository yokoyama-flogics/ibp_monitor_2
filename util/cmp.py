import sys

f1 = open('master.bayes', 'r')
f2 = open('1102b.bayes', 'r')

ct = 1
for line in f1.readlines():
    s1 = line.rstrip().split()
    sss = f2.readline()
    # print '@@' + sss + '##'
    if sss == '':
        sys.exit(0)

    s2 = sss.rstrip().split()

    # print s1, s2

    s11 = float(s1[1])
    s12 = float(s1[2])
    s21 = float(s2[1])
    s22 = float(s2[2])

    d1 = s21 - s11
    d2 = s22 - s12
    # if abs(d2) > 0.2:
    #    print "ERR", s1[0], s2[0], ct, d1, d2
    print ct, d1, d2

    ct += 1
