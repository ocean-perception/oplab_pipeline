import unittest
from auv_nav.tools.interpolate import interpolate

class TestTools(unittest.TestCase):
    def test_interpolate(self):
        x_query = 150.0
        x_lower = 100.0
        x_upper = 200.0
        y_lower = 100.0
        y_upper = 200.0
        y_query = interpolate(x_query, x_lower, x_upper, y_lower, y_upper)
        assert y_query == 150.0


if __name__ == '__main__':
    unittest.main()





