#!/usr/bin/env python3

import tensorflow as tf
import sys
import os
import logging
import argparse
from mnist import MNISTModel

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, default="basic_cnn", 
                        help="which model (basic_cnn, highway_cnn)")
    parser.add_argument("--batch-size", type=int, default=50, help="batch size")
    parser.add_argument("--batches", type=int, default=20000, help="total number of batches")
    parser.add_argument("--max-cpu-cores", type=int, default=1, help="how many cores to use at most")
    parser.add_argument("--device", default="cpu:0", help="which device to run on")
    parser.add_argument("--quiet", action='store_true', default=False, help="be quiet")

    args = parser.parse_args()

    loglevel = logging.ERROR if args.quiet else logging.INFO

    logging.basicConfig(format='%(levelname)s: %(message)s', level=loglevel)


    logging.info("Using this installation of TensorFlow: " + os.path.dirname(tf.__file__))

    with tf.device("/" + args.device):
        # initialize the model
        model = MNISTModel(
            batch_size=args.batch_size, 
            batches=args.batches, 
            model_type=args.model_type,
            quiet=args.quiet)
    
        # start a TensorFlow session
        session = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=args.max_cpu_cores,
                             intra_op_parallelism_threads=args.max_cpu_cores))
    
        # train
        logging.info("Training started.")
        model.train(session)
        logging.info("Training finished.")
    
        # evaluate on the official test set
        print(model.eval(session)[0])

if __name__ == "__main__":
    main(sys.argv)
