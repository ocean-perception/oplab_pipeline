from auv_nav.tools.interpolate import interpolate
from auv_nav.sensors import BodyVelocity


def test_interpolate():
    x_query = 150.0
    x_lower = 100.0
    x_upper = 200.0
    y_lower = 100.0
    y_upper = 200.0
    y_query = interpolate(x_query, x_lower, x_upper, y_lower, y_upper)
    assert y_query == 150.0


def test_BodyVelocity():
    b = BodyVelocity()
    autosub_data = {
        "eTime": [1574950320],
        "Vnorth0": [10.0],  # mm/s
        "Veast0": [10.0],
        "Vdown0": [10.0],
        "Verr0": [0.01],
        "Verr0": [0.01],
        "Verr0": [0.01]
    }
    b.from_autosub(autosub_data, 0)
    assert b.x_velocity == -0.01 and b.valid()  # m/s
