import unittest
from compare_mpi_records import compare,parse
LOG='''cal_rc=0 width=1920 height=1088 bytes=3133440 stride=1920
init_rc=0
alloc_rc=0
mb_size=4096 bytes_ok=1 before=1 add_result=0 after=2
release1=0 release2=0 exit_rc=0
'''
class ComparatorTests(unittest.TestCase):
    def test_equal_valid(self):self.assertTrue(compare(LOG,LOG,0,0)['pass'])
    def test_status_mismatch_even_exit_zero(self):self.assertFalse(compare(LOG,LOG.replace('add_result=0','add_result=2'),0,0)['pass'])
    def test_both_wrong_rejected(self):self.assertFalse(compare(LOG.replace('after=2','after=1'),LOG.replace('after=2','after=1'),0,0)['pass'])
    def test_missing_record(self):
        with self.assertRaises(ValueError):parse(LOG.replace('alloc_rc=0\n',''))
    def test_duplicate_record(self):
        with self.assertRaises(ValueError):parse(LOG+'alloc_rc=0\n')
    def test_duplicate_field(self):
        with self.assertRaises(ValueError):parse(LOG.replace('after=2','after=2 after=2'))
    def test_malformed_number(self):
        with self.assertRaises(ValueError):parse(LOG.replace('after=2','after=x'))
    def test_missing_field(self):
        with self.assertRaises(ValueError):parse(LOG.replace(' after=2',''))
    def test_nonzero_exit(self):
        with self.assertRaises(ValueError):compare(LOG,LOG,0,11)
    def test_diagnostic_retained_but_not_contract(self):self.assertTrue(compare('vendor diagnostic\n'+LOG,LOG,0,0)['pass'])
if __name__=='__main__':unittest.main()
