import os
import csv
import random

INPUT_DATASET = '/media/alejandro/DATA_SSD/temporary/SKU110K_fixed'
OUTPUT_DATASET = '/media/alejandro/DATA_SSD/temporary/SKU110K_fixed_100'
SAMPLES_NO = 100

os.system('mkdir -p ' + '"' + OUTPUT_DATASET + '"')
os.system('mkdir -p ' + '"' + os.path.join(OUTPUT_DATASET, 'annotations') + '"')
os.system('mkdir -p ' + '"' + os.path.join(OUTPUT_DATASET, 'images') + '"')

number_test = 0
for subdir, dirs, files in os.walk(INPUT_DATASET):
    for filename in files:
        filepath = subdir + os.sep + filename
        if "test" in filepath:
            number_test += 1

print("INFO: Number of test images " + str(number_test))

os.system('touch ' + '"' + os.path.join(OUTPUT_DATASET, 'annotations/annotations_test.csv') + '"')
<<<<<<< HEAD
<<<<<<< HEAD
os.system('touch ' + '"' + os.path.join(OUTPUT_DATASET, 'annotations/images.lst') + '"')

=======
>>>>>>> 034bc38ebd09ddd6f4aa5ad48516569e3479de39
=======
>>>>>>> 034bc38ebd09ddd6f4aa5ad48516569e3479de39

with open(os.path.join(OUTPUT_DATASET, 'annotations/annotations_test.csv'), 'a') as out_file:
    for idx in range(SAMPLES_NO):
        img_no = random.randint(0, number_test)
        print("INFO: image " + str(img_no))
        csv_file = open(os.path.join(INPUT_DATASET, 'annotations/annotations_test.csv'))
        csv_reader = csv.reader(csv_file, delimiter=',')
<<<<<<< HEAD
<<<<<<< HEAD
        # Compatibility with PoolNet
        with open(os.path.join(OUTPUT_DATASET, 'annotations/images.lst'), 'a') as out_file2:
            out_file2.write('test_' + str(img_no) + '.jpg' + '\n')

=======
>>>>>>> 034bc38ebd09ddd6f4aa5ad48516569e3479de39
=======
>>>>>>> 034bc38ebd09ddd6f4aa5ad48516569e3479de39
        for row in csv_reader:
            if row[0] == 'test_' + str(img_no) + '.jpg':
                # Write annotation
                out_file.write(', '.join(row) + '\n')
        os.system("cp " + os.path.join(INPUT_DATASET, 'images/test_' + str(img_no) + ".jpg") + " " +
                  os.path.join(OUTPUT_DATASET, 'images/test_' + str(img_no) + ".jpg"))


