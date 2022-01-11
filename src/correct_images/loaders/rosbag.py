# fmt: off
ROSBAG_IS_AVAILABLE = False
try:
    import rosbag
    from cv_bridge import CvBridge
    ROSBAG_IS_AVAILABLE = True
except ImportError:
    pass
# fmt: on


def loader(img_file, bagfile, src_bit=None):
    """Load an image from a ROS bagfile"""
    if not ROSBAG_IS_AVAILABLE:
        raise ImportError("ROS bagfile support is not available.")
    bridge = CvBridge()
    with rosbag.Bag(str(bagfile)) as bag:
        for topic, msg, t in bag.read_messages(topics=[str(img_file)]):
            if topic == str(img_file):
                return bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    return None
