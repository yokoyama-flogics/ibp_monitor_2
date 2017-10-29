import os
import sys
import unittest

# Set Python search path to the parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from sigretr import *

class TestSigRetr(unittest.TestCase):
    def setUp(self):
        import shutil

        os.environ['HOME'] = '.'

        # Change directory to unit tests exist (Not to remove bm2.cfg in the
        # top directory)
        self.origdir = os.getcwd()
        os.chdir(os.path.join(os.path.dirname(__file__), '.'))
        shutil.copyfile('test_config.cfg', 'bm2.cfg')

    def tearDown(self):
        os.remove('bm2.cfg')
        os.chdir(self.origdir)

    def test_sigretr_main_noarg(self):
        sys.argv = ['']
        with self.assertRaises(SystemExit):
            main()
        self.assertRegexpMatches(sys.stderr.getvalue(), 'too few arguments')

    def test_sigretr_main_datestr1(self):
        sys.argv = ['', 'error', '1', 'foo.wav']
        with self.assertRaises(SystemExit):
            main()
        self.assertRegexpMatches(sys.stderr.getvalue(), 'Illegal date')

    def test_sigretr_main_datestr2(self):
        sys.argv = ['', '20171028a', '1', 'foo.wav']
        with self.assertRaises(SystemExit):
            main()
        self.assertRegexpMatches(sys.stderr.getvalue(), 'Illegal date')

    def test_sigretr_main_line(self):
        sys.argv = ['', '20171028', '0', 'foo.wav']
        with self.assertRaises(SystemExit):
            main()
        self.assertRegexpMatches(sys.stderr.getvalue(), 'Illegal line')

    def test_sigretr_main_noinput(self):
        sys.argv = ['', '20171027', '1', 'foo.wav']
        with self.assertRaisesRegexp(IOError, 'No such file or directory'):
            main()

    def test_sigretr_main_success(self):
        sys.argv = ['', '-d', '20171028', '1', 'foo.wav']
        main()
        self.assertEqual(len(open('foo.wav').read()), 640044)
        self.assertEqual(open('foo.wav').read(),
                         open('sigretr_test.wav').read())
        os.remove('foo.wav')
        self.assertEqual(sys.stdout.getvalue(),
"""Current index:  1509148799 198927       16000 00:00:00 4U1UN  14MHz
Next index:  1509148809 200225      656000 00:00:10 VE8AT  14MHz
Start: 198927 16000
End: 200225 656000
Updated end_time:  10200225.0
# of samples:  160000
true_start_pos:  3272
""")

if __name__ == "__main__":
    unittest.main(buffer=True)
