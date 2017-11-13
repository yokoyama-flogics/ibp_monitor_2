import gd

lborder = 9
rborder = 31
tborder = 18
bborder = 8 
cwidth = 5
cheight = 5
sskip = 17

def hsv_to_rgb(h, s, v):
	import math
	hi = int(math.floor(h / 60.0)) % 6
	f = h / 60.0 - hi
	p = v * (1.0 - s)
	q = v * (1.0 - f * s)
	t = v * (1.0 - (1.0 - f) * s)
	p = int(math.floor(p * 255.0 + 0.5))
	q = int(math.floor(q * 255.0 + 0.5))
	t = int(math.floor(t * 255.0 + 0.5))
	v = int(math.floor(v * 255.0 + 0.5))
	if hi == 0:
		r = v
		g = t
		b = p
	elif hi == 1:
		r = q
		g = v
		b = p
	elif hi == 2:
		r = p
		g = v
		b = t
	elif hi == 3:
		r = p
		g = q
		b = v
	elif hi == 4:
		r = t
		g = p
		b = v
	elif hi == 5:
		r = v
		g = p
		b = q
	else:
		r = 0
		g = 0
		b = 0
	# print r, g, b
	return (r, g, b)

def getindex(sn, pp, bias):
	sni = int(sn / 3)
	if sni < 0:
		sni = 0
	if sni > 5:
		sni = 5

	ppi = int((pp - 0.5) / 0.1)
	if ppi < 0:
		ppi = 0
	if ppi > 4:
		ppi = 4

	if bias < -5.0:
		bii = 0
	elif bias < -3.0:
		bii = 1
	elif bias < -1.0:
		bii = 2
	elif bias < 1.0:
		bii = 3
	elif bias < 3.0:
		bii = 4
	elif bias < 5.0:
		bii = 5
	else:
		bii = 6

	return (sni, ppi, bii)

def iminit(im, colidx):
	import os
	global nosig
	global white
	global black
	global darkgray
	callsigns = (
		'4U1UN (United Nations)',
		'VE8AT (Canada)',
		'W6WX (United States)',
		'KH6RS (Hawaii)',
		'ZL6B (New Zealand)',
		'VK6RBP (Australia)',
		'JA2IGY (Japan)',
		'RR9O (Russia)',
		'VR2B (Hong Kong)',
		'4S7B (Sri Lanka)',
		'ZS6DN (South Africa)',
		'5Z4B (Kenya)',
		'4X6TU (Israel)',
		'OH2B (Finland)',
		'CS3B (Madeira)',
		'LU4AA (Argentina)',
		'OA4B (Peru)',
		'YV5B (Venezuela)')

	for sni in range(6):
		for ppi in range(5):
			for bii in range(7):
				colidx[(sni, ppi, bii)] = \
					im.colorAllocate(hsv_to_rgb(
						120 + (bii - 3) * 20,
						ppi * 0.2 + 0.2,
						sni * 0.1 + 0.5))

	nosig          = im.colorAllocate(hsv_to_rgb(120, 0.0, 0.5))
	white          = im.colorAllocate(hsv_to_rgb(  0, 0.0, 1.0))

	bg             = im.colorAllocate(hsv_to_rgb(  0, 0.0, 0.7))
	black = im.colorAllocate((0, 0, 0))
	darkgray = im.colorAllocate(hsv_to_rgb(0, 0, 0.2))
	# im.colorTransparent(-1)
	im.fill((0, 0), bg)
	for i in range(18):
		x1 = lborder - 1
		y1 = tborder - 1 + i * (cheight * 5 + sskip)
		x2 = lborder + cwidth * 96 - 1
		y2 = tborder + i * (cheight * 5 + sskip) + cheight * 5 - 1
		im.rectangle((x1, y1), (x2, y2), darkgray)
		im.string(gd.gdFontMediumBold, (x1, y1 - 13), callsigns[i], black)
		for j in range(12, 95, 12):
			im.line((x1 + cwidth * j, y1), (x1 + cwidth * j, y2), \
				darkgray)
		im.string(gd.gdFontTiny, (x2 + 6, y1 - 1), "10m", black)
		im.string(gd.gdFontTiny, (x2 + 6, y1 - 1 + cheight * 4), "20m", black)
			

	# for i in range(0, 50, 5):
	#	im.filledRectangle((0, 50 - i), (5, 50 - i + 5), colidx[i])

def imputmark(im, tindex, bindex, sindex, colidx):
	import math
	global nosig
	global white

	x = lborder + tindex * cwidth
	y = tborder + sindex * (cheight * 5 + sskip) + bindex * cheight
	# print x, y, col, dB
	# print (x, y), (x + 8, y + 8), colidx[col]
	if tindex % 4 == 3:
		split = 2
	else:
		split = 0
	im.filledRectangle((x, y), (x + cwidth - 2, y + cheight - 2), colidx)

def main():
	import gd
	import math
	import os
	import string
	import sys

	if len(sys.argv) != 2:
		sys.stderr.write("usage: genimg.py <date>\n")
		sys.exit(2)

	date = sys.argv[1]

	atoi = string.atoi
	bindex_tbl = {'14': 4, '18': 3, '21': 2, '24': 1, '28': 0}
	sindex_tbl = {
		"4U1UN":	0,
		"VE8AT":	1,
		"W6WX":		2,
		"KH6WO":	3,
		"ZL6B":		4,
		"VK6RBP":	5,
		"JA2IGY":	6,
		"RR9O":		7,
		"VR2B":		8,
		"4S7B":		9,
		"ZS6DN":	10,
		"5Z4B":		11,
		"4X6TU":	12,
		"OH2B":         13,
		"CS3B":         14,
		"LU4AA":        15,
		"OA4B":         16,
		"YV5B":         17}

	im = gd.image((lborder + rborder + cwidth * 96 - 1, \
		       tborder + bborder + 18 * (cheight * 5 + sskip) - sskip))
	colidx = {}
	iminit(im, colidx)

	f = open(os.environ["EXEC_DIR"] + "/db/ibprec_" + date + ".bayes", "r")
	for line in f:
# 00:04:00 VK6RBP 18MHz SN:  -1.4 Bias:   4.2 PProb: 0.00
		s = string.split(line)
		if len(s) < 9:
			continue

		(time, call, freq) = s[0:3]
# 00:00:00 4U1UN  14MHz SN:  -0.3 Bias: -246 Ct: 2 IF:  105 Z: 105 ZSN:   1.9
		sn = string.atof(s[4])
		bias = string.atof(s[6])
		pp = string.atof(s[8])

		tindex = string.split(time, ':')
		tindex = (atoi(tindex[0]) * 3600 + atoi(tindex[1]) * 60 + \
			  atoi(tindex[2])) / (15 * 60)
		bindex = bindex_tbl[string.split(freq, 'M')[0]]
		sindex = sindex_tbl[call]
		# print time, call, freq, sn, bias, tindex, bindex, \
		#       sindex

		found = (pp >= 0.5)
		if found:
			# print "found", pp
			imputmark(im, tindex, bindex, sindex, colidx[
				getindex(sn, pp, bias)])
		else:
			imputmark(im, tindex, bindex, sindex, nosig)

	fimg = open(date + ".png", "w")
	im.writePng(fimg)
	f.close()
	fimg.close()

if __name__ == "__main__":
	main()
