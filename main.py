#!/usr/bin/env python3

import tensorflow as tf
import sys
import os
import logging
import argparse
from mnist import MNISTModel

def main(args):
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=50, help="batch size")
    parser.add_argument("--batches", type=int, default=20000, help="total number of batches")
    parser.add_argument("--drop-out-prob", type=float, default=0.5, help="drop-out probability, 0 to disable")
    parser.add_argument("--device", default="cpu:0", help="device to run on")
    parser.add_argument("--max-cpu-cores", type=int, default=1, help="how many cores to use at most")

    args = parser.parse_args()

    logging.info("Using this installation of TensorFlow: " + os.path.dirname(tf.__file__))

    model = MNISTModel(
        batch_size=args.batch_size, 
        batches=args.batches, 
        dropout_prob=args.drop_out_prob)

    session = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=args.max_cpu_cores,
                         intra_op_parallelism_threads=args.max_cpu_cores))

    model.train(session)
    print("ACCURACY=" + str(model.eval(session)))

if __name__ == "__main__":
    main(sys.argv)