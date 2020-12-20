from PIL import Image
import argparse
import os

def impath(string):
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError(string)

parser = argparse.ArgumentParser(description='crop an image')
parser.add_argument('--imagepath', type=impath)
parser.add_argument('--size', type=int, help='size of cropped square image')

args = parser.parse_args()

if __name__ == '__main__':
    im = Image.open(args.imagepath) 
    if args.size > im.size[0] or args.size > im.size[1]:
        raise Exception("Image cannot be cropped to larger size")

    imcropped = im.crop((0, 0, args.size, args.size)) 

    imcropped.save(os.path.split("cropped_"+args.imagepath)[1])