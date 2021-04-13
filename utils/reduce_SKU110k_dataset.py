import os

INPUT_DATASET = '/media/alejandro/TOSHIBA EXT/ale/'
OUTPUT_DATASET = '/media/alejandro/TOSHIBA EXT/ale/'

os.system('mkdir -p ' + '"' + OUTPUT_DATASET + '"')

for subdir, dirs, files in os.walk(INPUT_DATASET):
    for filename in files:
        filepath = subdir + os.sep + filename
        if filepath.endswith(".jpg") or filepath.endswith(".png"):
            pass