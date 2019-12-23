from __future__ import division

import os
import sys
import time
import argparse
from datetime import datetime

import numpy as np

import pickle

from model import faster_rcnn

from model import config, data_generators

from model.parser import get_data
import model.roi_helpers as roi_helpers

from keras import backend as K
from keras.utils import generic_utils

sys.setrecursionlimit(10000)

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='Object Detection')
parser.add_argument("-p", "--path", default=None, help="path to annotation file")
parser.add_argument("--save_dir", default="./save", help="path to save directory")
parser.add_argument('--n_epochs', default=10, type=int, metavar='N',
                    help='number of epochs')
parser.add_argument('--n_iters', default=10, type=int, metavar='N',
                    help='number of iterations')
parser.add_argument('--horizontal_flips', action='store_true',
                    help='augument with horizontal flips (Default:False)')
parser.add_argument('--vertical_flips', action='store_true',
                    help='augument with horizontal flips (Default:False)')
parser.add_argument('--rot_90', action='store_true',
                    help='augument with 90 degree rotations (Default:False)')

def main():
    args = parser.parse_args()
    time_stamp = "{0:%Y%m%d-%H%M%S}".format(datetime.now())
    save_name = os.path.join(args.save_dir, "train_{}".format(time_stamp))

    if not(os.path.isdir(args.save_dir)):
        os.makedirs(args.save_dir)
    if args.path == None:
        raise OSError("path to annotation file must be required.")
    C = config.Config()
    C.config_filename = save_name + "_config.pickle"
    C.model_path = save_name + "_model.hdf5"
    C.use_horizontal_flips = bool(args.horizontal_flips)
    C.use_vertical_flips = bool(args.vertical_flips)
    C.rot_90 = bool(args.rot_90)
    all_imgs, classes_count, class_mapping = get_data(args.path)
    C.class_mapping = class_mapping

    with open(C.config_filename, 'wb') as config_f:
        pickle.dump(C,config_f)
        print("-------------------------------")
        print('path to config file : {}'.format(C.config_filename))
        print("-------------------------------")

    train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
    val_imgs = [s for s in all_imgs if s['imageset'] == 'test']

    data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, K.image_data_format(), mode='train')
    data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, K.image_data_format(), mode='val')

    model_rpn, model_classifier, model_all = faster_rcnn.get_model(C, classes_count)

    losses = np.zeros((args.n_iters, 5))
    rpn_accuracy_rpn_monitor, rpn_accuracy_for_epoch = [], []

    best_loss = np.Inf

    with open('out.csv', 'w') as f:
        f.write('Accuracy,RPN classifier,RPN regression,Detector classifier,Detector regression,Total')
        f.write('\t')

    iter_num = 0

    t0 = start_time = time.time()
    print(model_all.summary())
    try:
        for epoch_num in range(args.n_epochs):
            progbar = generic_utils.Progbar(args.n_iters)
            print('Epoch {}/{}'.format(epoch_num + 1, args.n_epochs))

            while True:
                try:
                    if len(rpn_accuracy_rpn_monitor) == args.n_iters and C.verbose:
                        mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
                        rpn_accuracy_rpn_monitor = []
                        print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, args.n_iters))
                        if mean_overlapping_bboxes == 0:
                            print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')
                    X, Y, img_data = next(data_gen_train)
                    loss_rpn = model_rpn.train_on_batch(X, Y)
                    P_rpn = model_rpn.predict_on_batch(X)
                    R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_data_format(), use_regr=True, overlap_thresh=0.7, max_boxes=300)

                    # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
                    X2, Y1, Y2 = roi_helpers.calc_iou(R, img_data, C, class_mapping)

                    neg_samples = np.where(Y1[0, :, -1] == 1)
                    pos_samples = np.where(Y1[0, :, -1] == 0)
                    if len(neg_samples) > 0:
                        neg_samples = neg_samples[0]
                    else:
                        neg_samples = []

                    if len(pos_samples) > 0:
                        pos_samples = pos_samples[0]
                    else:
                        pos_samples = []

                    rpn_accuracy_rpn_monitor.append(len(pos_samples))
                    rpn_accuracy_for_epoch.append((len(pos_samples)))
                    if len(pos_samples) < C.num_rois//2:
                        selected_pos_samples = pos_samples.tolist()
                    else:
                        selected_pos_samples = np.random.choice(pos_samples, C.num_rois//2, replace=False).tolist()
                    try:
                        selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
                    except:
                        selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()

                    sel_samples = selected_pos_samples + selected_neg_samples

                    loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

                    if iter_num == args.n_iters:
                        loss_rpn_cls = np.mean(losses[:, 0])
                        loss_rpn_regr = np.mean(losses[:, 1])
                        loss_class_cls = np.mean(losses[:, 2])
                        loss_class_regr = np.mean(losses[:, 3])
                        class_acc = np.mean(losses[:, 4])

                        mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                        rpn_accuracy_for_epoch = []

                        if C.verbose:
                            print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
                            print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                            print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                            print('Loss RPN regression: {}'.format(loss_rpn_regr))
                            print('Loss Detector classifier: {}'.format(loss_class_cls))
                            print('Loss Detector regression: {}'.format(loss_class_regr))
                            print('Elapsed time: {}[s]'.format(time.time() - start_time))

                        target_text_file = open('out.csv', 'a')
                        target_text_file.write('{},{},{},{},{},{}'.format(class_acc, loss_rpn_cls,
                                                loss_rpn_regr, loss_class_cls, loss_class_regr,
                                                loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr))
                        target_text_file.write('\t')

                        curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                        iter_num = 0
                        start_time = time.time()

                        if curr_loss < best_loss:
                            if C.verbose:
                                print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
                            best_loss = curr_loss
                            model_all.save_weights(C.model_path)
                        break

                    losses[iter_num, 0] = loss_rpn[1]
                    losses[iter_num, 1] = loss_rpn[2]
                    losses[iter_num, 2] = loss_class[1]
                    losses[iter_num, 3] = loss_class[2]
                    losses[iter_num, 4] = loss_class[3]
                    iter_num += 1

                    progbar.update(iter_num, [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                                              ('detector_cls', np.mean(losses[:iter_num, 2])), ('detector_regr', np.mean(losses[:iter_num, 3]))])

                except Exception as e:
                    print('Exception: {}'.format(e))
                    continue

    except KeyboardInterrupt:
        t1 = time.time()
        print('\nIt took {:.2f}s'.format(t1-t0))
        sys.exit('Keyboard Interrupt')

    print("training is done")
    print("-------------------------------")
    print('path to config file : {}'.format(C.config_filename))
    print("-------------------------------")

if __name__ == '__main__':
    main()
