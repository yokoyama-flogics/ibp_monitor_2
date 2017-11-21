import sys

f1 = open('ibprec_20171114.log', 'r')
f2 = open('20171114_charex.log', 'r')

ct = 1
for line in f1.readlines():
    s1 = line.rstrip().split()
    s2 = f2.readline().rstrip().split()

    # print s1, s2

    time = int(s2[0])
    sn_diff = float(s2[4]) - float(s1[4]) 
    ct_diff = int(s2[6]) - int(s1[8])
    if_diff = int(s2[7]) - int(s1[10])
    bg_diff = float(s2[8]) - float(s1[11])

    # if abs(sn_diff) > 0.1:
    #     print time, sn_diff

    # if abs(ct_diff) > 1:
    #     print time, ct_diff, s1, s2

    # if abs(if_diff) > 1:
    #     print time, if_diff, s1, s2

    if abs(bg_diff) > 0.1:
        print time, bg_diff, s1, s2

    continue




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
