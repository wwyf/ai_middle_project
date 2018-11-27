from typing import Tuple, Mapping
import json
import time
import itertools
import numpy as np


class LabelTrainer:
    def __init__(self,
                 data: np.ndarray,
                 data_labels: np.ndarray,
                 labels: Tuple,
                 parameters: Mapping,
                 config_function,
                 train_function,
                 predict_function):
        self.data = data
        self.data_labels = data_labels
        # create a map for labels
        self.labels = {}
        for i in range(0, len(labels)):
            self.labels[labels[i]] = i
        self.parameters = parameters
        self.config_function = config_function
        self.train_function = train_function
        self.predict_function = predict_function
        self.best_parameters = {}

    def auto_train(self,
                   validation_ratio=0.2,
                   validation_split_times=5,
                   maximize="precision",
                   print_info=False) -> Mapping:
        """
        @param validation_ratio: ratio of validation set size
        @param validation_split_times: times to split dataset
        @param maximize: attribute to be maximized, can be "precision", "recall" or "f1"
        @param print_info: print train info or not
        @return best parameter set
        """
        t = 0
        maximize = maximize.lower()
        if maximize == "precision":
            t = 0
        elif maximize == "recall":
            t = 1
        elif maximize == "f1":
            t = 2
        validation_num = int(self.data.shape[0] * validation_ratio)
        validation_sets = []
        train_sets = []
        indexes = np.arange(0, self.data.shape[0])
        for i in range(0, validation_split_times):
            np.random.shuffle(indexes)
            validation_sets.append(indexes[0:validation_num])
            train_sets.append(indexes[validation_num:self.data.shape[0]])

        # train the model
        param_combinations = itertools.product(*self.parameters.values())

        if print_info:
            print(" >>> starting to determine parameters <<<")
            print("{:>10}{:>15}{:>15}{:>25}{:>15}".format("round", "time(s)", maximize, "(p, r, f1)", "parameters"))

        global_round = 0
        global_start = time.time()
        best_parameters = ()
        best_score = 0.0
        for p_comb in param_combinations:
            avg_score = 0.0
            for train_round in range(0, validation_split_times):
                self.config_function(self._convert_to_map(tuple(self.parameters.keys()), p_comb))

                train_start = time.time()
                self.train_function(self.data[train_sets[train_round]],
                                    self.data_labels[train_sets[train_round]])
                result = self.predict_function(self.data[validation_sets[train_round]])
                train_end = time.time()

                # gather results
                confusion_mat = np.zeros([len(self.labels), len(self.labels)])
                for i in range(0, validation_num):
                    true_label = self.data_labels[validation_sets[train_round][i]]
                    predict_idx = self.labels[result[i]]
                    true_idx = self.labels[true_label]
                    if result[i] == true_label:
                        confusion_mat[predict_idx][predict_idx] += 1
                    else:
                        confusion_mat[predict_idx][true_idx] += 1

                # evaluate score
                score = self._evaluate_score(confusion_matrix=confusion_mat)
                avg_score += score[t]
                if print_info:
                    print("{:>10s}{:>15.2f}{:>15.3f}     {:>6.3f} {:>6.3f} {:>6.3f}   | {:s}".format(
                        "%d-%d" % (global_round, train_round),
                        train_end - train_start,
                        score[t], score[0], score[1], score[2],
                        str(self._convert_to_map(tuple(self.parameters), p_comb))))
                global_round += 1

            avg_score /= validation_split_times
            if avg_score > best_score:
                best_score = avg_score
                best_parameters = p_comb

        global_end = time.time()
        self.best_parameters = self._convert_to_map(tuple(self.parameters), best_parameters)
        if print_info:
            print(" Optimization time: {:.2f}s, Total rounds: {:d}".format(
                global_end - global_start, global_round))
            print(" Best score: {:.3f}".format(best_score))
            print(" Best parameters: {:s}".format(str(self.best_parameters)))

        return self.best_parameters

    def config_model(self) -> None:
        """
        @brief configure a model to use the best parameters
        """
        self.config_function(self.best_parameters)

    def train_model(self) -> None:
        """
        @brief train the model
        """
        self.train_function(self.data, self.data_labels)

    def save_parameters(self, path) -> None:
        """
        @brief save the best parameters
        @param path: path of output file
        """
        with open(path, "w") as output_file:
            json.dump(self.best_parameters, output_file)

    def load_parameters(self, path) -> None:
        """
        @brief load the best parameters
        @param path: path of input file
        """
        with open(path, "r") as input_file:
            self.best_parameters = json.load(path)

    @staticmethod
    def _convert_to_map(keys: Tuple, values: Tuple) -> Mapping:
        result = {}
        for i in range(0, len(keys)):
            result[keys[i]] = values[i]
        return result

    @staticmethod
    def _evaluate_score(confusion_matrix: np.ndarray):
        """
        :param confusion_matrix: 6x6 confusion matrix
        :return: average precision, average recall, average f1 score
        """
        EPS = 0.0000001
        size = confusion_matrix.shape[0]
        tp = np.zeros(size, np.float32)
        fp = np.zeros(size, np.float32)
        fn = np.zeros(size, np.float32)
        for i in range(0, size):
            tp[i] = confusion_matrix[i][i]
            fp[i] = confusion_matrix[i].sum() - tp[i]
            fn[i] = confusion_matrix[:, i].sum() - tp[i]
        p = tp + fn
        precision = tp / (tp + fp + EPS)
        recall = tp / (p + EPS)
        f_1 = 2 * precision * recall / ((precision + recall) + EPS)
        return precision.mean(), recall.mean(), f_1.mean()
