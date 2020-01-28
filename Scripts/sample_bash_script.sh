#!/bin/bash
location_of_py_file="Main.py" #full path to Main.py
mode="test" #train/validate/test
output_dir="UI_test" #output file directory
batchsize=24 #default 24 (required)
architecture="P1" #P1/P2/P3/P4/PC1/PC2/PC3/PC4
feature="PTEN" #feature to predict; see README
epoch=100000 #default 100000 (required)
modeltoload="../path/to/trained/model" #only for validate and test
imagefile="../path/to/image/to/test" #only for test
resolution=20 #default None (required)
BMI=35 #required for test
age=59 #required for test
label_file="../sample_label.csv" #required for train and validate
split_file="..sample_sep_file.csv" #required for train and validate (default None)


python3 ${location_of_py_file} \
--mode ${mode} \
--out_dir ${output_dir} \
--batchsize ${batchsize} \
--architecture ${architecture} \
--feature ${feature} \
--epoch ${epoch} \
--modeltoload ${modeltoload} \
--imagefile ${imagefile} \
--resolution ${resolution} \
--BMI ${BMI} \
--age ${age} \
--label_file ${label_file} \
--split_file ${split_file}

