import unittest
from auv_nav.parsers.parse_phins import PhinsTimestamp


class TestPhinsTimestamp(unittest.TestCase):
    def setUp(self):
        date = [2018, 11, 19]
        timezone = 10.0    # in hours
        timeoffset = 1.0   # in seconds
        self.t = PhinsTimestamp(date, timezone, timeoffset)

    def test_epoch_from_offset(self):
        hour = 8
        mins = 30
        secs = 15
        msec = 23
        epoch = self.t.get(hour, mins, secs, msec)
        self.assertEqual(epoch, 1542580216.023, 'Time conversion is wrong')
        
    def test_timestamp_from_phins(self):
        line = ['TIME__', ' ', '083015.023']
        epoch = self.t.epoch_timestamp_from_phins(line)
        self.assertEqual(epoch, 1542580216.023, 'Time conversion is wrong')


if __name__ == '__main__':
    unittest.main()





