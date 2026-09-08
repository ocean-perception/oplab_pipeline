"""Generate sample of processed images to evaluate correct images

Usage:
example_image_tiles.py im_dir

positional arguments:
    im_dir          Path to processed dive folder
    
optional arguments:
    -h, --help      show this help message and exit
    
"""



#  import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from glob import glob
from PIL import Image
import argparse

def example_image_tiles(im_dir,photo_start=None,photo_end=None):
    for files in glob(im_dir+'/image/*/corrected_*/'):
        print('-'.join(files.split('/')[files.split('/').index('processed')+1:]))
        im_list = glob(files+'*.png')
        if photo_start is not None and photo_end is not None:
            im_list = im_list[int(photo_start):int(photo_end)+1]
        else:
            if photo_start is not None:
                im_list = im_list[int(photo_start):]
            if photo_end is not None:
                im_list = im_list[:int(photo_end)+1]
        img = Image.open(im_list[0])
        width,height = img.size
        scale_factor = width*height/1000000
        img.close()
        random.shuffle(im_list)
        fig = plt.figure(figsize=[20,height/(width/20)])

        for n in range(len(im_list))[:100]:
            img = Image.open(im_list[n])
            img.thumbnail(([i/(scale_factor) for i in img.size]))
            # img = cv2.imread(im_list[n])
            # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

            plt.subplot(10,10,n+1)
            plt.imshow(img)
            plt.axis('off')
            img.close()

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        plt.tight_layout()
        plt.savefig('/'.join(im_list[0].split('/')[:-2])+'/'+im_list[0].split('/')[-2]+'-sampled_images.png')
        plt.close()
        print('image saved at '+'/'.join(im_list[0].split('/')[:-2])+'/'+im_list[0].split('/')[-2]+'-sampled_images.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("im_dir", help="Path to processed dive folder")
    parser.add_argument("photo_start", nargs='?', default=None, help="Photo index to start with")
    parser.add_argument("photo_end", nargs='?', default=None, help="Photo index to end with")
    args = parser.parse_args()
    example_image_tiles(
        args.im_dir,
        args.photo_start,
        args.photo_end
    )

