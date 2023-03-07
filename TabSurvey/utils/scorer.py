from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, log_loss, roc_auc_score, \
    confusion_matrix
import numpy as np


def get_scorer(args):
    if args.objective == "regression":
        return RegScorer()
    elif args.objective == "classification":
        return ClassScorer()
    elif args.objective == "binary" or args.objective == "binary_f1":
        return BinScorer()
    else:
        raise NotImplementedError("No scorer for \"" + args.objective + "\" implemented")


class Scorer:
    """
        y_true: (n_samples,)
        y_prediction: (n_samples,) - predicted classes
        y_probabilities: (n_samples, n_classes) - probabilities of the classes (summing to 1)
    """

    def eval(self, y_true, y_prediction, y_probabilities):
        raise NotImplementedError("Has be implemented in the sub class")

    def get_results(self):
        raise NotImplementedError("Has be implemented in the sub class")

    def get_objective_result(self):
        raise NotImplementedError("Has be implemented in the sub class")


class RegScorer(Scorer):

    def __init__(self):
        self.mses = []
        self.r2s = []
        self.f1s = []
        self.accs = []

    # y_probabilities is None for Regression
    def eval(self, y_true, y_prediction, y_probabilities):
        mse = mean_squared_error(y_true, y_prediction)
        r2 = r2_score(y_true, y_prediction)
        y_prediction_ = np.array([int(i>0.5) for i in y_prediction])
        print(f'np.all(y_prediction_==0) : {np.all(y_prediction_ == 0)}')
        print(f'np.all(y_prediction_==1) : {np.all(y_prediction_ == 1)}')
        print(f"np.sum(np.array(y_prediction)) : {np.sum(np.array(y_prediction_))}")
        print(f"np.array(y_prediction).size : {np.array(y_prediction_).size}")
        f1 = f1_score(y_true, y_prediction_, average="binary")
        acc = accuracy_score(y_true, y_prediction_)
        self.mses.append(mse)
        self.r2s.append(r2)
        self.f1s.append(f1)
        self.accs.append(acc)

        return {"MSE": mse, "R2": r2,"F1 score":f1,"accs":acc}

    def get_results(self):
        mse_mean = np.mean(self.mses)
        mse_std = np.std(self.mses)


        r2_mean = np.mean(self.r2s)
        r2_std = np.std(self.r2s)

        f1_mean = np.mean(self.f1s)
        f1_std = np.std(self.f1s)

        acc_mean = np.mean(self.accs)
        acc_std = np.std(self.accs)

        return {"MSE - mean": mse_mean,
                "MSE - std": mse_std,
                "R2 - mean": r2_mean,
                "R2 - std": r2_std,
                "f1 - mean": f1_mean,
                "f1 - std": f1_std,
                "acc - mean": acc_mean,
                "acc - std": acc_std
                }

    def get_objective_result(self):
        return np.mean(self.mses)


class ClassScorer(Scorer):

    def __init__(self):
        self.loglosses = []
        self.aucs = []
        self.accs = []
        self.f1s = []

    def eval(self, y_true, y_prediction, y_probabilities):
        logloss = log_loss(y_true, y_probabilities)
        # auc = roc_auc_score(y_true, y_probabilities, multi_class='ovr')
        # auc = roc_auc_score(y_true, y_probabilities, multi_class='ovo', average="macro")

        acc = accuracy_score(y_true, y_prediction)
        f1 = f1_score(y_true, y_prediction, average="binary")  # use here macro or weighted?
        print(f'np.all(y_prediction==0) : {np.all(np.array(y_prediction) == 0)}')
        print(f'np.all(y_prediction==1) : {np.all(np.array(y_prediction) == 1)}')
        print(f"np.sum(np.array(y_prediction)) : {np.sum(np.array(y_prediction))}")
        print(f"np.array(y_prediction).size : {np.array(y_prediction).size}")
        self.loglosses.append(logloss)
        # self.aucs.append(auc)
        self.accs.append(acc)
        self.f1s.append(f1)

        # return {"Log Loss": logloss, "AUC": auc, "Accuracy": acc, "F1 score": f1}
        return {"Log Loss": logloss, "Accuracy": acc, "F1 score": f1}

    def get_results(self):
        logloss_mean = np.mean(self.loglosses)
        logloss_std = np.std(self.loglosses)

        auc_mean = np.mean(self.aucs)
        auc_std = np.std(self.aucs)

        acc_mean = np.mean(self.accs)
        acc_std = np.std(self.accs)

        f1_mean = np.mean(self.f1s)
        f1_std = np.std(self.f1s)

        return {"Log Loss - mean": logloss_mean,
                "Log Loss - std": logloss_std,
                "AUC - mean": auc_mean,
                "AUC - std": auc_std,
                "Accuracy - mean": acc_mean,
                "Accuracy - std": acc_std,
                "F1 score - mean": f1_mean,
                "F1 score - std": f1_std,
                "loglosses":self.loglosses,
                "aucs":self.aucs,
                "accs":self.accs,
                "F1 score":self.f1s
                }

    def get_objective_result(self):
        return np.mean(self.loglosses)


class BinScorer(Scorer):

    def __init__(self):
        self.loglosses = []
        self.aucs = []
        self.accs = []
        self.f1s = []
        self.accu_1 = []
        self.accu_2 = []

    def eval(self, y_true, y_prediction, y_probabilities):
        y_true = y_true.reshape((-1,))
        print(np.array(y_prediction))
        print(np.array(y_true))
        print(y_prediction.shape)
        print(y_true.shape)
        logloss = log_loss(y_true, y_probabilities)
        auc = roc_auc_score(y_true, y_probabilities[:, 1])

        acc = accuracy_score(y_true, y_prediction)
        f1 = f1_score(y_true, y_prediction, average="binary")  # use here macro or weighted?
        print(f'np.all(y_prediction==0) : {np.all(np.array(y_prediction) == 0)}')
        print(f'np.all(y_prediction==1) : {np.all(np.array(y_prediction) == 1)}')
        print(f"np.sum(np.array(y_prediction)) : {np.sum(np.array(y_prediction))}")
        print(f"np.array(y_prediction).size : {np.array(y_prediction).size}")
        self.loglosses.append(logloss)
        self.aucs.append(auc)
        self.accs.append(acc)
        self.f1s.append(f1)
        tn, fp, fn, tp = confusion_matrix(y_true, y_prediction).ravel()
        # print('0：不涨 ， 1：涨')
        print('tn, fp, fn, tp', tn, fp, fn, tp)
        #
        print(f'test中为1的比例 : {y_true.sum() / len(y_true)}')
        print(f'test中为0的比例 : {(1 - y_true).sum() / len(y_true)}')

        # error_in_test = mean_squared_error(y_test_hat, np.array(testing_df[target_fea]).reshape(-1, 1))
        accu_1 = tp / (tp + fp)
        accu_2 = tp / (tp + fn)
        print(f'查准率 - 预测为1 且实际为1 ，看涨的准确率: {accu_1}')
        print(f'查全率 - 实际为1，预测为1 : {accu_2}')
        self.accu_1.append(accu_1)
        self.accu_2.append(accu_2)
        return {"Log Loss": logloss, "AUC": auc, "Accuracy": acc, "F1 score": f1, "accu_1": accu_1, "accu_2": accu_2}

    def get_results(self):
        logloss_mean = np.mean(self.loglosses)
        logloss_std = np.std(self.loglosses)

        auc_mean = np.mean(self.aucs)
        auc_std = np.std(self.aucs)

        acc_mean = np.mean(self.accs)
        acc_std = np.std(self.accs)

        f1_mean = np.mean(self.f1s)
        f1_std = np.std(self.f1s)

        accu_1_mean = np.mean(self.accu_1)
        accu_1_std = np.std(self.accu_1)

        accu_2_mean = np.mean(self.accu_2)
        accu_2_std = np.std(self.accu_2)

        return {"Log Loss - mean": logloss_mean,
                "Log Loss - std": logloss_std,
                "AUC - mean": auc_mean,
                "AUC - std": auc_std,
                "Accuracy - mean": acc_mean,
                "Accuracy - std": acc_std,
                "F1 score - mean": f1_mean,
                "F1 score - std": f1_std,
                "accu_1 score - mean": accu_1_mean,
                "accu_1 score - std": accu_1_std,
                "accu_2 score - mean": accu_2_mean,
                "accu_2 score - std": accu_2_std,
                "loglosses": self.loglosses,
                "aucs": self.aucs,
                "accs": self.accs,
                "F1": self.f1s}



    def get_objective_result(self):
        return np.mean(self.aucs)
