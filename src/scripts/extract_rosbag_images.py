import argparse
from pathlib import Path

import cv2
import rosbag
from cv_bridge import CvBridge


def main():
    """Extract a folder of images from a rosbag."""
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("bag_file", nargs="+", help="Input ROS bag.")
    parser.add_argument("output_dir", help="Output directory.")
    parser.add_argument("image_topic", help="Image topic.")

    args = parser.parse_args()

    print(
        "Extract images from %s on topic %s into %s"
        % (args.bag_file, args.image_topic, args.output_dir)
    )

    for bag_file in args.bag_file:
        bag = rosbag.Bag(bag_file, "r")
        bridge = CvBridge()
        count = 0
        for topic, msg, t in bag.read_messages(topics=[args.image_topic]):
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            stamp = str(msg.header.stamp)
            fname = Path(args.output_dir)
            if not fname.exists():
                fname.mkdir(parents=True)
            fname = fname / ("frame" + stamp[:-9] + "." + stamp[-9:] + ".png")
            cv2.imwrite(str(fname), cv_img)
            print("Wrote image %s" % str(fname))
            count += 1
        bag.close()
    return
