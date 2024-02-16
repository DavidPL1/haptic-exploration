# Copyright 2015 Yale University - Grablab
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:\
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# Modified to work with Python 3 by Sebastian Castro, 2020
# Modified with argparse to allow for command line arguments and paths with ros by David Leins, 2024

import rospy
from rospkg import RosPack

from haptic_exploration import mujoco_config

import os
import os.path as osp
import json
from urllib.request import Request, urlopen
import argparse

parser = argparse.ArgumentParser(description="Download YCB models")
parser.add_argument("--output-directory", "-o", type=str, default=osp.join(RosPack().get_path('haptic_exploration'), "ycb_downloads"), help="Output directory for YCB models")
parser.add_argument("--objects-to-download", '-d', nargs="*", required=True, help="List of objects to download. Use 'all' to download all objects, 'demo' to download a small set of objects, or a list of objects separated by spaces.")
parser.add_argument("--files-to-download", '-f', nargs="*", default=["google_16k"], help="List of files to download. Use 'all' to download all files, or a list of files separated by spaces.")
parser.add_argument("--no-extract", action="store_true", help="Do not extract downloaded files")


# Define a list of objects to download from
# http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/

# You can edit this list to only download certain kinds of files.
# 'berkeley_rgbd' contains all of the depth maps and images from the Carmines.
# 'berkeley_rgb_highres' contains all of the high-res images from the Canon cameras.
# 'berkeley_processed' contains all of the segmented point clouds and textured meshes.
# 'google_16k' contains google meshes with 16k vertices.
# 'google_64k' contains google meshes with 64k vertices.
# 'google_512k' contains google meshes with 512k vertices.

def fetch_objects(url):
    """ Fetches the object information before download """
    response = urlopen(url)
    html = response.read()
    objects = json.loads(html)
    return objects["objects"]


def download_file(url, filename):
    """ Downloads files from a given URL """
    u = urlopen(url)
    f = open(filename, "wb")
    file_size = int(u.getheader("Content-Length"))
    print("Downloading: {} ({} MB)".format(filename, file_size / 1000000.0))

    file_size_dl = 0
    block_sz = 65536
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl / 1000000.0, file_size_dl * 100. / file_size)
        status = status + chr(8) * (len(status) + 1)
        print(status)
    f.close()


def tgz_url(object, type):
    """ Get the TGZ file URL for a particular object and dataset type """
    if type in ["berkeley_rgbd", "berkeley_rgb_highres"]:
        return base_url + "berkeley/{object}/{object}_{type}.tgz".format(object=object, type=type)
    elif type in ["berkeley_processed"]:
        return base_url + "berkeley/{object}/{object}_berkeley_meshes.tgz".format(object=object, type=type)
    else:
        return base_url + "google/{object}_{type}.tgz".format(object=object, type=type)


def extract_tgz(filename, dir):
    """ Extract a TGZ file """
    tar_command = "tar -xzf {filename} -C {dir}".format(filename=filename, dir=dir)
    os.system(tar_command)
    os.remove(filename)


def check_url(url):
    """ Check the validity of a URL """
    try:
        request = Request(url)
        request.get_method = lambda: 'HEAD'
        response = urlopen(request)
        return True
    except Exception as e:
        return False


if __name__ == "__main__":
    rospy.init_node('ycb_downloader', anonymous=True)
    base_url = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/"
    objects_url = "https://ycb-benchmarks.s3.amazonaws.com/data/objects.json"

    args = parser.parse_args()
    output_directory = args.output_directory
    objects_to_download = args.objects_to_download
    files_to_download = args.files_to_download
    extract = not args.no_extract

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if files_to_download == ["all"]:
        files_to_download = ["berkeley_rgbd", "berkeley_rgb_highres", "berkeley_processed", "google_16k", "google_64k", "google_512k"]
    if objects_to_download == ["demo"]:
        objects_to_download = ["001_chips_can",
                            "002_master_chef_can",
                            "003_cracker_box",
                            "004_sugar_box"]

    base_ycb_dir = osp.join(RosPack().get_path('haptic_exploration'), 'assets', 'meshes', 'ycb')
    available_objects = os.listdir(base_ycb_dir)
    available_objects = [x for x in available_objects if osp.isdir(osp.join(base_ycb_dir, x))]
    available_objects = [x for x in available_objects if x in mujoco_config.ycb_objects.values()]

    # Grab all the object information
    objects = fetch_objects(objects_url)

    # Download each object for all objects and types specified
    for object in objects:
        if objects_to_download == ["all"] or object in objects_to_download:
            if object in available_objects:
                rospy.loginfo(f"Object {object} already available. Skipping download")
                continue
            for file_type in files_to_download:
                url = tgz_url(object, file_type)
                if not check_url(url):
                    continue
                filename = "{path}/{object}_{file_type}.tgz".format(
                    path=output_directory,
                    object=object,
                    file_type=file_type)
                download_file(url, filename)
                if extract:
                    extract_tgz(filename, osp.join(RosPack().get_path('haptic_exploration'), 'assets', 'meshes', 'ycb'))
