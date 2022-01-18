from auv_nav.sensors import ros_stamp_to_epoch

# fmt: off
ROSBAG_IS_AVAILABLE = False
try:
    import rosbag
    from cv_bridge import CvBridge
    ROSBAG_IS_AVAILABLE = True
except ImportError:
    pass
# fmt: on

TOL = 1e-3


def loader(img_timestamp, img_topic, bagfile_list, tz_offset_s, src_bit=None):
    """Load an image from a ROS bagfile"""
    if not ROSBAG_IS_AVAILABLE:
        raise ImportError("ROS bagfile support is not available.")
    bridge = CvBridge()
    img_timestamp = float(img_timestamp)
    # print("Loading image at", img_timestamp)
    for bagfile in bagfile_list:
        with rosbag.Bag(str(bagfile)) as bag:
            for topic, msg, t in bag.read_messages(topics=[str(img_topic)]):
                epoch_timestamp = ros_stamp_to_epoch(msg.header.stamp) - tz_offset_s
                # print(abs(epoch_timestamp - img_timestamp))
                if abs(epoch_timestamp - img_timestamp) < TOL:
                    # print("Image found!")
                    return bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    # print("KKKKKKKKKKKKKKKKK NOT FOUND")
    return None