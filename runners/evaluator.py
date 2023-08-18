import numpy as np

from vp_estimation_with_prior_gravity.evaluation import (
    get_vp_accuracy, get_vp_detection_ratio, get_recall_AUC, pose_auc)


class Evaluator(object):
    def __init__(self, vp_metrics=True, use_rot=True, uncalibrated=True):
        self.vp_metrics = vp_metrics
        self.use_rot = use_rot
        self.uncalibrated = uncalibrated
        self.scores = {}
        self.tmp_scores = {} # for multirun
        self.types = []
        if self.vp_metrics:
            self.types += ["vp_accuracy", "vp_detection_ratio", "rec", "auc", "auc_std"]
        if self.use_rot:
            self.types.append("rot_error")
        if self.uncalibrated:
            self.types.append("f_error")
        for t in self.types:
            self.scores[t] = {}
            self.tmp_scores[t] = {}
        self.methods = [] # existing methods

        # hyperparameters
        self.deg_thresholds = np.arange(1, 11)

    def add_method(self, m):
        if m not in self.methods:
            self.methods.append(m)
            for t in self.types:
                self.scores[t][m] = []
                self.tmp_scores[t][m] = []

    def eval_scores(self, scores, method, vp, gt_vp, K):
        self.add_method(method)
        scores["vp_accuracy"][method].append(get_vp_accuracy(gt_vp, vp, K))
        scores["vp_detection_ratio"][method].append(get_vp_detection_ratio(gt_vp, vp, K, self.deg_thresholds))
        recall, auc = get_recall_AUC(gt_vp, vp, K)
        scores["rec"][method].append(recall)
        scores["auc"][method].append(auc)
        scores["auc_std"][method].append(0)

    # interfaces
    def add_entry(self, method, vp, gt_vp, K):
        self.eval_scores(self.scores, method, vp, gt_vp, K)

    def add_entry_multirun(self, method, vp, gt_vp, K):
        self.eval_scores(self.tmp_scores, method, vp, gt_vp, K)

    def add_rot_error(self, method, rot_error):
        if method not in self.methods:
            raise ValueError("No data for the method {0} is added.".format(method))
        self.scores["rot_error"][method].append(rot_error)

    def add_rot_error_multirun(self, method, rot_error):
        if method not in self.methods:
            raise ValueError("No data for the method {0} is added.".format(method))
        self.tmp_scores["rot_error"][method].append(rot_error)

    def add_f_error(self, method, f_error):
        if method not in self.methods:
            raise ValueError("No data for the method {0} is added.".format(method))
        self.scores["f_error"][method].append(f_error)

    def add_f_error_multirun(self, method, f_error):
        if method not in self.methods:
            raise ValueError("No data for the method {0} is added.".format(method))
        self.tmp_scores["f_error"][method].append(f_error)

    def end_multirun(self, method):
        if method not in self.methods:
            raise ValueError("No data for the method {0} is added.".format(method))
        # if len(self.tmp_scores["vp_accuracy"][method]) == 0:
        #     raise ValueError("No multirun data is added")

        if self.vp_metrics:
            num_runs = len(self.tmp_scores["vp_accuracy"][method])
            self.scores["vp_accuracy"][method].append(np.median(np.stack(self.tmp_scores["vp_accuracy"][method], axis=0), axis=0))
            self.scores["vp_detection_ratio"][method].append(np.median(np.stack(self.tmp_scores["vp_detection_ratio"][method], axis=0), axis=0))
            self.scores["rec"][method].append(self.tmp_scores["rec"][method][np.argsort(self.tmp_scores["auc"][method])[num_runs // 2]])
            self.scores["auc"][method].append(np.mean(self.tmp_scores["auc"][method]))
            self.scores["auc_std"][method].append(np.std(self.tmp_scores["auc"][method]))
        if self.use_rot:
            self.scores["rot_error"][method].append(np.median(self.tmp_scores["rot_error"][method]))
        if self.uncalibrated:
            self.scores["f_error"][method].append(np.median(self.tmp_scores["f_error"][method]))

        # clear cache
        for t in self.types:
            self.tmp_scores[t][method] = []

    def aggregate(self):
        report = {}
        if self.vp_metrics:
            report["vp_accuracy"] = {m: np.stack(val, axis=0).mean(axis=0) for m, val in self.scores["vp_accuracy"].items()}
            report["vp_detection_ratio"] = {m: np.stack(val, axis=0).mean(axis=0) for m, val in self.scores["vp_detection_ratio"].items()}
            report["rec"] = {m: np.stack(val, axis=0).mean(axis=0) for m, val in self.scores["rec"].items()}
            report["auc"] = {m: np.stack(val, axis=0).mean(axis=0) for m, val in self.scores["auc"].items()}
            report["auc_std"] = {m: np.stack(val, axis=0).mean(axis=0) for m, val in self.scores["auc_std"].items()}
        if self.use_rot:
            report["rot_auc"] = {m: 100 * np.r_[pose_auc(val, thresholds=[5, 10, 20])] for m, val in self.scores["rot_error"].items()}
            report["rot_error"] = {m: np.median(val) for m, val in self.scores["rot_error"].items()}
        if self.uncalibrated:
            report["f_error"] = {m: np.median(val) for m, val in self.scores["f_error"].items()}
        return report

    def save_report(self, fname, method):
        report = self.aggregate()
        res = {k: v[method] for k, v in report.items()}
        np.savez(fname, **res)

    def report(self):
        report = self.aggregate()

        if self.use_rot:
            # Rotation estimation
            print("[REPORT] Median rotation error: ")
            for m in self.methods:
                print("{0}: {1:.2f}".format(m, report["rot_error"][m]))
            print()
            print("[REPORT] Rotation AUC at 5 / 10 / 20 deg error thresholds:")
            for m in self.methods:
                print("{0}: {1:.1f} / {2:.1f} / {3:.1f}".format(m, report["rot_auc"][m][0], report["rot_auc"][m][1], report["rot_auc"][m][2]))
            print()

        if self.vp_metrics:
            # VP accuracy
            # TODO: vp consistency
            print("[REPORT] VP error:")
            for m in self.methods:
                print("{0}: {1:.2f}".format(m, report["vp_accuracy"][m]))
            print()

            # VP AUC
            print("[REPORT] VP Recall AUC:")
            for m in self.methods:
                print("{0}: {1:.2f}, std = {2:.2f}".format(m, report["auc"][m], report["auc_std"][m]))
            print()

        if self.uncalibrated:
            # Focal length estimation
            print("[REPORT] Median focal length error: ")
            for m in self.methods:
                print("{0}: {1:.3f}".format(m, report["f_error"][m]))
