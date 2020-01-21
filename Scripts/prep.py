"""
Input preparation functions; GUI input; tfrecords preparation; weight calculation.

Created on 01/21/2020

@author: RH
"""
import os
import sys
import pandas as pd
import tensorflow as tf
import cv2
import data_input
import numpy as np
import argparse
import easygui


dirr = sys.argv[1]  # output directory
bs = sys.argv[2]  # batch size
bs = int(bs)
md = sys.argv[3]  # architecture to use
pdmd = sys.argv[4]  # feature to predict

try:
    ep = sys.argv[5]  # epochs to train
    ep = int(ep)
except IndexError:
    ep = 100

try:
    sup = sys.argv[6]  # fusion mode
except IndexError:
    sup = False

if pdmd == 'subtype':
    classes = 4
else:
    classes = 2

print('Input config:')
print(dirr, bs, md, pdmd, ep, sup)

# input image dimension
INPUT_DIM = [bs, 299, 299, 3]
# hyper parameters
HYPERPARAMS = {
    "batch_size": bs,
    "dropout": 0.3,
    "learning_rate": 1E-4,
    "classes": classes,
    "sup": sup
}

# paths to directories
img_dir = '../tiles/'
LOG_DIR = "../Results/{}".format(dirr)
METAGRAPH_DIR = "../Results/{}".format(dirr)
data_dir = "../Results/{}/data".format(dirr)
out_dir = "../Results/{}/out".format(dirr)


def input_handler():
    parser = argparse.ArgumentParser(description="Parse some arguments")
    parser.add_argument('--gui', type=str, choices=['True', 'False'], default='False')
    parser.add_argument('--mode', type=str, choices=['train', 'validate', 'test'], default='train')
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--batchsize', type=int, default=24)
    parser.add_argument('--architecture', type=str, default="P1")
    parser.add_argument('--feature', type=str, default=None)
    parser.add_argument('--epoch', type=float, default=np.inf)
    parser.add_argument('--modeltoload', type=str, default=None)
    parser.add_argument('--path_to_modeltoload', type=str, default=None)
    parser.add_argument('--imagefile', type=str, default=None)

    args = parser.parse_args()
    gui = args.gui == 'True'
    if gui:
        # Box1
        msg = "Hello! Hola! Bonjour! Ciao! I'm Panoptes GUI." \
              "By clicking continue, you agree with my terms and conditions. " \
              "Do you want to continue?"
        title = "Please Confirm"
        if easygui.ccbox(msg, title):  # show a Continue/Cancel dialog
            pass  # user chose Continue
        else:  # user chose Cancel
            sys.exit(0)
        # Box2
        msg = "How may I help you? (Train a new model/Validate a model/Test to predict a new sample)"
        choices = ['train', 'validate', 'test']
        mode = easygui.buttonbox(msg, choices=choices)
        # Box3
        msg = "Okay! Where would you like your results to go? (Name a folder under 'Results' directory)"
        out_dir = easygui.enterbox(msg)
        # Box4
        msg = "What would you like to predict today?"
        title = "Select a feature to predict"
        choices = ["histology", "subtype", "subtype_POLE", "subtype_MSI", "subtype_CNV-L", "subtype_CNV-H", "ARID1A",
                   "ATM", "BRCA2", "CTCF", "CTNNB1", "FAT1", "FBXW7", "FGFR2", "JAK1", "KRAS", "MTOR", "PIK3CA",
                   "PIK3R1", "PPP2R1A", "PTEN", "RPL22", "TP53", "ZFHX3"]
        feature = easygui.choicebox(msg, title, choices)
        # Box5
        msg = "Which architecture do you want to use?"
        title = "Select an architecture"
        choices = ["P1", "P2", "P3", "P4", "PC1", "PC2", "PC3", "PC4"]
        architecture = easygui.choicebox(msg, title, choices)
        # Box6&7
        if mode == "validate":
            # Box6
            title = "Enter your information"
            msg = "I need some more information to proceed. " \
                    "Please make sure that the model to be loaded is of the same architecture you chose."
            fieldNames = ["trained model name", "full path to trained model"]
            fieldValues = []  # we start with blanks for the values
            fieldValues = easygui.multenterbox(msg, title, fieldNames)
            # make sure that none of the fields was left blank
            while 1:
                if fieldValues is None: break
                errmsg = ""
                for i in range(len(fieldNames)):
                    if fieldValues[i].strip() == "":
                        errmsg = errmsg + ('"%s" is a required field.\n\n' % fieldNames[i])
                if errmsg == "": break  # no problems found
                fieldValues = easygui.multenterbox(errmsg, title, fieldNames, fieldValues)
            modeltoload = fieldValues[0]
            path_to_modeltoload = fieldValues[1]
            imagefile = None
        elif mode == "test":
            # Box6
            title = "Enter your information"
            msg = "I need some more information to proceed. " \
                    "Please make sure that the model to be loaded is of the same architecture you chose."
            fieldNames = ["trained model name", "full path to trained model"]
            fieldValues = []  # we start with blanks for the values
            fieldValues = easygui.multenterbox(msg, title, fieldNames)
            # make sure that none of the fields was left blank
            while 1:
                if fieldValues is None: break
                errmsg = ""
                for i in range(len(fieldNames)):
                    if fieldValues[i].strip() == "":
                        errmsg = errmsg + ('"%s" is a required field.\n\n' % fieldNames[i])
                if errmsg == "": break  # no problems found
                fieldValues = easygui.multenterbox(errmsg, title, fieldNames, fieldValues)
            modeltoload = fieldValues[0]
            path_to_modeltoload = fieldValues[1]
            # Box7
            msg = "Select the slide you want to predict?"
            title = "Select a slide in the 'images' folder."
            choices = [f for f in os.listdir("../images")]
            imagefile = easygui.choicebox(msg, title, choices)
        else:
            modeltoload = None
            path_to_modeltoload = None
            imagefile = None
        # Box8
        msg = "Almost there! Do you agree with our default batch size and max epoch number?" \
              "Batch_size = 24; max epoch number = infinity"
        title = "Please Confirm"
        if easygui.ccbox(msg, title):  # show a Continue/Cancel dialog
            batchsize = 24
            epoch = np.inf
            pass  # user chose Continue
        else:  # user chose Cancel
            # Box9
            msg = "Please enter your choice (integer only)"
            title = "Enter your choice"
            fieldNames = ["batch size", "max epoch"]
            fieldValues = []  # we start with blanks for the values
            fieldValues = easygui.multenterbox(msg, title, fieldNames)
            # make sure that none of the fields was left blank
            while 1:
                if fieldValues is None: break
                errmsg = ""
                for i in range(len(fieldNames)):
                    if fieldValues[i].strip() == "":
                        errmsg = errmsg + ('"%s" is a required field.\n\n' % fieldNames[i])
                if errmsg == "": break  # no problems found
                fieldValues = easygui.multenterbox(errmsg, title, fieldNames, fieldValues)
            batchsize = int(fieldValues[0])
            epoch = int(fieldValues[1])
    else:
        mode = args.mode
        out_dir = args.out_dir
        batchsize = args.batchsize
        architecture = args.architecture
        feature = args.feature
        epoch = args.epoch
        modeltoload = args.modeltoload
        path_to_modeltoload = args.path_to_modeltoload
        imagefile = args.imagefile

    return mode, out_dir, batchsize, architecture, feature, epoch, modeltoload, path_to_modeltoload, imagefile

# count numbers of training and testing images
def counters(totlist_dir, cls):
    trlist = pd.read_csv(totlist_dir + '/tr_sample.csv', header=0)
    telist = pd.read_csv(totlist_dir + '/te_sample.csv', header=0)
    valist = pd.read_csv(totlist_dir + '/va_sample.csv', header=0)
    trcc = len(trlist['label'])
    tecc = len(telist['label'])
    vacc = len(valist['label'])
    weigh = []
    for i in range(cls):
        ccct = len(trlist.loc[trlist['label'] == i])+len(valist.loc[valist['label'] == i])\
               + len(telist.loc[telist['label'] == i])
        wt = ((trcc+tecc+vacc)/cls)/ccct
        weigh.append(wt)
    weigh = tf.constant(weigh)
    return trcc, tecc, vacc, weigh


# read images
def load_image(addr):
    img = cv2.imread(addr)
    img = img.astype(np.float32)
    return img


# used for tfrecord float generation
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


# used for tfrecord labels generation
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# used for tfrecord images generation
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# loading images for dictionaries and generate tfrecords
def loader(totlist_dir, ds, data_dir):
    if ds == 'train':
        slist = pd.read_csv(totlist_dir + '/tr_sample.csv', header=0)
    elif ds == 'validation':
        slist = pd.read_csv(totlist_dir + '/va_sample.csv', header=0)
    elif ds == 'test':
        slist = pd.read_csv(totlist_dir + '/te_sample.csv', header=0)
    else:
        slist = pd.read_csv(totlist_dir + '/te_sample.csv', header=0)
    imlista = slist['L0path'].values.tolist()
    imlistb = slist['L1path'].values.tolist()
    imlistc = slist['L2path'].values.tolist()
    lblist = slist['label'].values.tolist()
    wtlist = slist['BMI'].values.tolist()
    aglist = slist['age'].values.tolist()
    filename = data_dir + '/' + ds + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(filename)
    for i in range(len(lblist)):
        try:
            # Load the image
            imga = load_image(imlista[i])
            imgb = load_image(imlistb[i])
            imgc = load_image(imlistc[i])
            label = lblist[i]
            wt = wtlist[i]
            ag = aglist[i]
            # Create a feature
            feature = {ds + '/label': _int64_feature(label),
                       ds + '/BMI': _float_feature(wt),
                       ds + '/age': _float_feature(ag),
                       ds + '/imageL0': _bytes_feature(tf.compat.as_bytes(imga.tostring())),
                       ds + '/imageL1': _bytes_feature(tf.compat.as_bytes(imgb.tostring())),
                       ds + '/imageL2': _bytes_feature(tf.compat.as_bytes(imgc.tostring()))}
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())
        except AttributeError:
            print('Error image: ' + imlista[i] + '~' + imlistb[i] + '~' + imlistc[i])
            pass
    writer.close()


# load tfrecords and prepare datasets
def tfreloader(mode, ep, bs, cls, ctr, cte, cva, data_dir):
    filename = data_dir + '/' + mode + '.tfrecords'
    if mode == 'train':
        ct = ctr
    elif mode == 'test':
        ct = cte
    else:
        ct = cva

    datasets = data_input.DataSet(bs, ct, ep=ep, cls=cls, mode=mode, filename=filename)

    return datasets