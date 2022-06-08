from auv_nav.sensors import ros_stamp_to_epoch
from oplab import Console

# fmt: off
ROSBAG_IS_AVAILABLE = False
try:
    import rosbag
    import rospy
    from cv_bridge import CvBridge
    ROSBAG_IS_AVAILABLE = True
except ImportError:
    pass
# fmt: on

TOL = 5e-3
SEARCH_TOL = 0.5


def loader(img_timestamp, img_topic, bagfile_list, tz_offset_s, src_bit=8):
    """Load an image from a ROS bagfile"""
    if not ROSBAG_IS_AVAILABLE:
        raise ImportError("ROS bagfile support is not available.")
    bridge = CvBridge()
    img_timestamp = float(img_timestamp)
    # Bagfile filtering timestamps
    start_time = rospy.Time.from_sec(img_timestamp - SEARCH_TOL + tz_offset_s)
    end_time = rospy.Time.from_sec(img_timestamp + SEARCH_TOL + tz_offset_s)
    for bagfile in bagfile_list:
        with rosbag.Bag(str(bagfile)) as bag:
            for topic, msg, t in bag.read_messages(
                topics=[str(img_topic)],
                start_time=start_time,
                end_time=end_time,
            ):
                epoch_timestamp = ros_stamp_to_epoch(msg.header.stamp) - tz_offset_s
                # print(abs(epoch_timestamp - img_timestamp))
                if abs(epoch_timestamp - img_timestamp) < TOL:
                    # print("Image found at", t.to_sec())
                    image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                    image_float = image.astype(float) * 2 ** (-src_bit)
                    return image_float
    Console.warn("Image", img_timestamp, "not found in bagfile list.")
    return None
