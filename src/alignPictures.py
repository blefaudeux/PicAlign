#!/usr/bin/python

# alignPictures.py

# Picture alignment using OpenCV

# Usage: python alignPictures.py <image_directory> optional: <output_directory> <start_num>,<end_num>
# ffmpeg -r 15 -b 1800 -i %4d.JPG -i testSong.mp3 test1800.avi

import sys
import os
import TrackImage
from multiprocessing import Pool
from operator import itemgetter
from PIL import Image
# import argparse


def main():
    # Print usage if no args specified
    args = sys.argv[1:]
    if len(args) == 0:
        print('Usage: python alignPictures.py <image_directory> optional: <output_directory> <start_num>,<end_num>')
        return

    # Get input files, sort by last modified time
    files = sorted_images(args[0])

    # TODO: Streamline the arguments, use argparse
    if len(files) == 0:
        print('No jpg files found in ' + args[0])
        return

    if len(sys.argv) > 1:
        outdir = args[1]
    else:
        outdir = '.'

    start, end = 0, len(files) - 1
    if len(args) > 2:
        if ',' in args[2]:
            start, end = map(lambda x: int(x) - 1, args[2].split(','))
        else:
            start = int(args[2]) - 1

    files = files[start:end + 1]

    # - populate a task pool finding the interframe motion
    results = {}
    pool = Pool()
    first_frame = files[0][1]

    i = start
    for f in files:
        i += 1
        savename = os.path.join(outdir, '%04d.jpg' % i)
        print('Added to track pool: ' + f[1])
        results[f[1]] = pool.apply_async(TrackImage.run_track_image, (first_frame, f[1], savename))

    pool.close()
    pool.join()

    # - check the results, try to compensate motion through another reference if needed
    valid_frames = [item.get() for item in results.items()]
    print("Motion compensation results : \n{}".format(valid_frames))


def sorted_images(input_dir):
    files = []
    for dirpath, dirnames, filenames in os.walk(input_dir):
        for filename in filenames:
            if filename.upper().endswith('.JPG') or filename.upper().endswith('.JPEG'):
                file_path = os.path.join(dirpath, filename)
                files.append((get_image_date(file_path), file_path))

    # Sort by last modified, then by path
    files.sort(key=itemgetter(0, 1))
    return files


def get_image_date(file_path):
    """ This returns the date as a formatted string like yyyy:mm:dd hh:mm:ss. Which is good enough for sorting. """
    date_time_origin = 36867
    return Image.open(file_path)._getexif()[date_time_origin]


if __name__ == "__main__":
    main()
