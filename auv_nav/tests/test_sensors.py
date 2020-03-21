import unittest
from auv_nav.sensors import OutputFormat
from auv_nav.sensors import BodyVelocity

class TestOutputFormat(unittest.TestCase):
    def setUp(self):
        self.of = OutputFormat()

    def testComparison(self):
        other = OutputFormat()
        other.epoch_timestamp = 100
        self.of.epoch_timestamp = 10
        self.assertTrue(self.of < other)

class TestBodyVelocity(unittest.TestCase):
    def setUp(self):
        self.bv = BodyVelocity()

    def test_BodyVelocityFromAutosub(self):
        self.bv.clear()
        self.assertFalse(self.bv.valid())

        autosub_data = {
            "eTime": [1574950320],
            "Vnorth0": [10.0],  # mm/s
            "Veast0": [10.0],
            "Vdown0": [10.0],
            "Verr0": [0.01],
            "Verr0": [0.01],
            "Verr0": [0.01]
        }
        self.bv.from_autosub(autosub_data, 0)
        self.assertEqual(self.bv.x_velocity, -0.01,
                         'incorrect forward speed')
        self.assertTrue(self.bv.valid())

    def test_BodyVelocityFromPhins(self):
        self.bv.clear()
        self.assertFalse(self.bv.valid())

        phins_data = ['SPEED_', '', '0.2', '0.03', '0.1', '', '083015.23']
        self.bv.from_phins(phins_data)
        self.assertTrue(self.bv.valid)
        self.assertEqual(self.bv.x_velocity, 0.2, 'incorrect forward speed')
        self.assertEqual(self.bv.y_velocity, -0.03, 'incorrect forward speed')
        self.assertEqual(self.bv.z_velocity, -0.1, 'incorrect forward speed')



if __name__ == '__main__':
    unittest.main()





