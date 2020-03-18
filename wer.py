#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Zimeng Qiu <zimengq@andrew.cmu.edu>

"""
Simple implementation of WER
"""

import copy
import argparse
import numpy as np

from collections import defaultdict


def read_file(file):
    """
    Read utterance from text file
    :param file: input file path
    :return: lines - list[string]
    """
    with open(file, 'r') as f:
        lines = [line.split() for line in f.readlines()]

    return lines


class WER(object):
    """
    Abstract class for WER
    """
    def __init__(self, ref, hyp):
        """
        Initialization of WER
        :param ref: reference utterance
        :param hyp: hypothesis utterance
        """
        self.ref = ref
        self.hyp = hyp
        self.dist_matrix = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=np.uint8)
        self.optim_path = defaultdict(lambda: defaultdict(list))
        self.ref_mod = defaultdict(lambda: defaultdict(list))
        self.hyp_mod = defaultdict(lambda: defaultdict(list))

        # initialize DTW distance matrix for dynamic programming
        for i in range(len(ref) + 1):
            for j in range(len(hyp) + 1):
                if i == 0:
                    self.dist_matrix[0][j] = j
                elif j == 0:
                    self.dist_matrix[i][0] = i

    def distance(self):
        """
        Calculating DTW distance between reference and hypothesis utterances
        Using dynamic programming
        :return: DTW distance matrix
                 optimal operation path
                 modified reference utterance
                 modified hypothesis utterance
        """
        # traverse distance, optimal path and modified reference and hypothesis utterance matrix
        # with dynamic programming
        for i in range(1, self.dist_matrix.shape[0]):
            for j in range(1, self.dist_matrix.shape[1]):
                if self.ref[i - 1] == self.hyp[j - 1]:
                    # exact match
                    self.dist_matrix[i][j] = self.dist_matrix[i - 1][j - 1]
                    self.optim_path[i][j] = copy.deepcopy(self.optim_path[i - 1][j - 1])
                    self.ref_mod[i][j] = copy.deepcopy(self.ref_mod[i - 1][j - 1])
                    self.hyp_mod[i][j] = copy.deepcopy(self.hyp_mod[i - 1][j - 1])
                    max_len = max(3, len(self.ref[i - 1]))
                    self.optim_path[i][j].append(" " * max_len)
                    self.ref_mod[i][j].append(self.ref[i - 1] + " " * (max_len - len(self.ref[i - 1])))
                    self.hyp_mod[i][j].append(self.hyp[j - 1] + " " * (max_len - len(self.hyp[j - 1])))
                else:
                    # not match, choose insert/delete/substitute
                    cost = 1
                    self.dist_matrix[i][j] = cost + min(
                        self.dist_matrix[i - 1][j], self.dist_matrix[i - 1][j - 1], self.dist_matrix[i][j - 1])

                    cases = np.asarray([
                        self.dist_matrix[i - 1][j], self.dist_matrix[i - 1][j - 1], self.dist_matrix[i][j - 1]])

                    if np.argmin(cases) == 0:
                        # delete
                        self.optim_path[i][j] = copy.deepcopy(self.optim_path[i - 1][j])
                        self.ref_mod[i][j] = copy.deepcopy(self.ref_mod[i - 1][j])
                        self.hyp_mod[i][j] = copy.deepcopy(self.hyp_mod[i - 1][j])
                        max_len = max(3, len(self.ref[i - 1]))
                        self.optim_path[i][j].append("DEL" + " " * (max_len - 3))
                        self.ref_mod[i][j].append(self.ref[i - 1] + " " * (max_len - len(self.ref[i - 1])))
                        self.hyp_mod[i][j].append(" " * max_len)
                    elif np.argmin(cases) == 1:
                        # substitute
                        self.optim_path[i][j] = copy.deepcopy(self.optim_path[i - 1][j - 1])
                        self.ref_mod[i][j] = copy.deepcopy(self.ref_mod[i - 1][j - 1])
                        self.hyp_mod[i][j] = copy.deepcopy(self.hyp_mod[i - 1][j - 1])
                        max_len = max(len(self.ref[i - 1]), len(self.hyp[j - 1]), 3)
                        self.optim_path[i][j].append("SUB" + " " * (max_len - 3))
                        self.ref_mod[i][j].append(self.ref[i - 1] + (max_len - len(self.ref[i - 1])) * " ")
                        self.hyp_mod[i][j].append(self.hyp[j - 1] + (max_len - len(self.hyp[j - 1])) * " ")
                    elif np.argmin(cases) == 2:
                        # insert
                        self.optim_path[i][j] = copy.deepcopy(self.optim_path[i][j - 1])
                        self.ref_mod[i][j] = copy.deepcopy(self.ref_mod[i][j - 1])
                        self.hyp_mod[i][j] = copy.deepcopy(self.hyp_mod[i][j - 1])
                        max_len = max(3, len(self.hyp[j - 1]))
                        self.optim_path[i][j].append("INS" + " " * (max_len - 3))
                        self.ref_mod[i][j].append(" " * max_len)
                        self.hyp_mod[i][j].append(self.hyp[j - 1] + " " * (max_len - len(self.hyp[j - 1])))

        return self.dist_matrix, self.optim_path, self.ref_mod, self.hyp_mod


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('ref', help='reference text file', type=str)
    parser.add_argument('hyp', help='hypothesis text file', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    refs = read_file(args.ref)
    hyps = read_file(args.hyp)

    for ref, hyp in zip(refs, hyps):
        wer = WER(ref, hyp)
        distance, path, ref_mod, hyp_mod = wer.distance()
        print("REF: " + ' '.join(ref_mod[len(ref)][len(hyp)]))
        print("HYP: " + ' '.join(hyp_mod[len(ref)][len(hyp)]))
        print("     " + ' '.join(path[len(ref)][len(hyp)]))
        print("WER: {:.0f}%".format(100 * distance[len(ref)][len(hyp)] / len(ref)))


