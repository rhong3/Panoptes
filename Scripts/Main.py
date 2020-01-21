"""
Main method

Created on 01/21/2020

@author: RH
"""
import os
import sys
import tensorflow as tf
import pandas as pd
from openslide import OpenSlide
import matplotlib
matplotlib.use('Agg')
import prep
import cnn
import sample_prep


# main; trc is training image count; tec is testing image count; to_reload is the model to load; test or not
def main(trc, tec, vac, cls, weight, testset=None, to_reload=None, test=None):
    if test:  # restore for testing only
        m = cnn.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR,
                           meta_dir=LOG_DIR, model=md, weights=weight)
        print("Loaded! Ready for test!")
        if tec >= bs:
            THE = tfreloader('test', 1, bs, cls, trc, tec, vac)
            m.inference(THE, dirr, testset=testset, pmd=pdmd)
        else:
            print("Not enough testing images!")

    elif to_reload:  # restore for further training and testing
        m = cnn.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR,
                           meta_dir=LOG_DIR, model=md, weights=weight)
        print("Loaded! Restart training.")
        HE = tfreloader('train', ep, bs, cls, trc, tec, vac)
        VHE = tfreloader('validation', ep*100, bs, cls, trc, tec, vac)
        itt = int(trc * ep / bs)
        if trc <= 2 * bs or vac <= bs:
            print("Not enough training/validation images!")
        else:
            m.train(HE, VHE, trc, bs, pmd=pdmd, dirr=dirr, max_iter=itt, save=True, outdir=METAGRAPH_DIR)
        if tec >= bs:
            THE = tfreloader('test', 1, bs, cls, trc, tec, vac)
            m.inference(THE, dirr, testset=testset, pmd=pdmd)
        else:
            print("Not enough testing images!")

    else:  # train and test
        m = cnn.INCEPTION(INPUT_DIM, HYPERPARAMS, log_dir=LOG_DIR, model=md, weights=weight)
        print("Start a new training!")
        HE = tfreloader('train', ep, bs, cls, trc, tec, vac)
        VHE = tfreloader('validation', ep*100, bs, cls, trc, tec, vac)
        itt = int(trc*ep/bs)+1
        if trc <= 2 * bs or vac <= bs:
            print("Not enough training/validation images!")
        else:
            m.train(HE, VHE, trc, bs, pmd=pdmd, dirr=dirr, max_iter=itt, save=True, outdir=METAGRAPH_DIR)
        if tec >= bs:
            THE = tfreloader('test', 1, bs, cls, trc, tec, vac)
            m.inference(THE, dirr, testset=testset, pmd=pdmd)
        else:
            print("Not enough testing images!")


if __name__ == "__main__":
    tf.reset_default_graph()
    mode, out_dir, feature, architecture, modeltoload, path_to_modeltoload, \
    imagefile, batchsize, epoch, resolution = prep.input_handler()

    if architecture in ["PC1", "PC2", "PC3", "PC4"]:
        sup = True
    else:
        sup = False

    if feature == 'subtype':
        classes = 4
    else:
        classes = 2

    # input image dimension
    INPUT_DIM = [batchsize, 299, 299, 3]
    # hyper parameters
    HYPERPARAMS = {
        "batch_size": batchsize,
        "dropout": 0.3,
        "learning_rate": 1E-4,
        "classes": classes,
        "sup": sup
    }

    # paths to directories
    img_dir = '../tiles/'
    LOG_DIR = "../Results/{}".format(out_dir)
    METAGRAPH_DIR = "../Results/{}".format(out_dir)
    data_dir = "../Results/{}/data".format(out_dir)
    out_dir = "../Results/{}/out".format(out_dir)

    # make directories if not exist
    for DIR in (LOG_DIR, METAGRAPH_DIR, data_dir, out_dir):
        try:
            os.mkdir(DIR)
        except FileExistsError:
            pass

    if mode == "test":
        if feature == 'histology':
            pos_score = "Serous_score"
            neg_score = "Endometrioid_score"
        else:
            pos_score = "POS_score"
            neg_score = "NEG_score"


    # get counts of testing, validation, and training datasets;
    # if not exist, prepare testing and training datasets from sampling
    try:
        trc, tec, vac, weights = counters(data_dir, classes)
        trs = pd.read_csv(data_dir + '/tr_sample.csv', header=0)
        tes = pd.read_csv(data_dir+'/te_sample.csv', header=0)
        vas = pd.read_csv(data_dir+'/va_sample.csv', header=0)
    except FileNotFoundError:
        alll = sample_prep.big_image_sum(pmd=pdmd, path=img_dir)
        # trs, tes, vas = sample_prep.set_sep_secondary(alll, path=data_dir, cls=classes, pmd=pdmd, batchsize=bs)
        trs, tes, vas = sample_prep.set_sep_idp(alll, path=data_dir, cls=classes, batchsize=bs)
        trc, tec, vac, weights = counters(data_dir, classes)
        loader(data_dir, 'train')
        loader(data_dir, 'validation')
        loader(data_dir, 'test')
    # have trained model or not; train from scratch if not
    try:
        modeltoload = sys.argv[7]
        # test or not
        try:
            testmode = sys.argv[8]
            main(trc, tec, vac, classes, weights, testset=tes, to_reload=modeltoload, test=True)
        except IndexError:
            main(trc, tec, vac, classes, weights, testset=tes, to_reload=modeltoload)
    except IndexError:
        if not os.path.isfile(data_dir + '/test.tfrecords'):
            loader(data_dir, 'test')
        if not os.path.isfile(data_dir + '/train.tfrecords'):
            loader(data_dir, 'train')
        if not os.path.isfile(data_dir + '/validation.tfrecords'):
            loader(data_dir, 'validation')
        if sup:
            print("Using Fusion Mode!")
        main(trc, tec, vac, classes, weights, testset=tes)

