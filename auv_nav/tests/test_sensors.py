from auv_nav.tools.interpolate import interpolate

def test_interpolate():
    x_query = 150.0
    x_lower = 100.0
    x_upper = 200.0
    y_lower = 100.0
    y_upper = 200.0
    y_query = interpolate(x_query, x_lower, x_upper, y_lower, y_upper)
    assert y_query == 150.0


