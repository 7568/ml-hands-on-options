import logging
import sys

import optuna

from utils import logger_conf
from models import str2model
from utils.load_data_3d import load_data
from utils.scorer import get_scorer, BinScorer
from utils.timer import Timer
from utils.io_utils import save_results_to_file, save_hyperparameters_to_file, save_loss_to_file
from utils.parser import get_parser, get_given_parameters_parser

from sklearn.model_selection import KFold, StratifiedKFold  # , train_test_split


def cross_validation(model, X, y, args, save_model=False):
    # Record some statistics and metrics
    sc = get_scorer(args)
    train_timer = Timer()
    test_timer = Timer()

    if args.objective == "regression":
        kf = KFold(n_splits=args.num_splits, shuffle=args.shuffle, random_state=args.seed)
    elif args.objective == "classification" or args.objective == "binary":
        kf = StratifiedKFold(n_splits=args.num_splits, shuffle=args.shuffle, random_state=args.seed)
    else:
        raise NotImplementedError("Objective" + args.objective + "is not yet implemented.")

    for i, (train_index, test_index) in enumerate(kf.split(X, y)):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=args.seed)

        # Create a new unfitted version of the model
        curr_model = model.clone()
        print(f'args.use_gpu {args.use_gpu}')
        if args.use_gpu:
            import torch
            device = torch.device(f"cuda:{args.gpu_index}" if torch.cuda.is_available() else "cpu")
            curr_model.device = device
            print(f'curr_model.device {curr_model.device}')
        # Train model
        train_timer.start()
        loss_history, val_loss_history = curr_model.fit(X_train, y_train, X_test, y_test)  # X_val, y_val)
        train_timer.end()

        # Test model
        test_timer.start()
        curr_model.predict(X_test)
        test_timer.end()

        # Save model weights and the truth/prediction pairs for traceability
        curr_model.save_model_and_predictions(y_test, i)

        if save_model:
            save_loss_to_file(args, loss_history, "loss", extension=i)
            save_loss_to_file(args, val_loss_history, "val_loss", extension=i)

        # Compute scores on the output
        sc.eval(y_test, curr_model.predictions, curr_model.prediction_probabilities)

        print(sc.get_results())

    # Best run is saved to file
    if save_model:
        print("Results:", sc.get_results())
        print("Train time:", train_timer.get_average_time())
        print("Inference time:", test_timer.get_average_time())

        # Save the all statistics to a file
        save_results_to_file(args, sc.get_results(),
                             train_timer.get_average_time(), test_timer.get_average_time(),
                             model.params)

    # print("Finished cross validation")
    return sc, (train_timer.get_average_time(), test_timer.get_average_time())


def training_validation_testing(model, X, y, training_trading_dates, validation_trading_dates, testing_trading_dates,args, save_model=False):
    # Record some statistics and metrics
    sc = get_scorer(args)
    sc2 = BinScorer()
    test_timer = Timer()

    # for i, (train_index, test_index) in enumerate(kf.split(X, y)):

    X_train, X_validation, X_test = X['training'], X['validation'], X['testing']
    y_train, y_validation, y_test = y['training'], y['validation'], y['testing']

    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=args.seed)

    # Create a new unfitted version of the model
    curr_model = model.clone()

    # print(f'args.use_gpu {args.use_gpu}')
    # if args.use_gpu:
    #     import torch
    #     device = torch.device(f"cuda:{args.gpu_index}" if torch.cuda.is_available() else "cpu")
    #     curr_model.device = device
    #     print(f'curr_model.device {curr_model.device}')
    curr_model.blation_test_id=2

    test_timer.start()
    curr_model.set_testing(X_test,y_test,testing_trading_dates)
    curr_model.predict(X_test,testing_trading_dates)
    test_timer.end()
    sc.eval(curr_model.y_test,curr_model.predictions, curr_model.prediction_probabilities)


    print(sc.get_results())



class Objective(object):
    def __init__(self, args, model_name, X, y):
        # Save the model that will be trained
        self.model_name = model_name

        # Save the trainings data
        self.X = X
        self.y = y

        self.args = args

    def __call__(self, trial):
        # Define hyperparameters to optimize
        trial_params = self.model_name.define_trial_parameters(trial, self.args)

        # Create model
        model = self.model_name(trial_params, self.args)

        # Cross validate the chosen hyperparameters
        if not self.args.dataset == 'H_sh_300_options':
            sc, time = cross_validation(model, self.X, self.y, self.args)
        else:
            training_validation_testing(model, self.X, self.y, self.args)

        return sc.get_objective_result()


def main(args):
    print("Start hyperparameter optimization")
    X, y,training_trading_dates, validation_trading_dates, testing_trading_dates = load_data(args)

    model_name = str2model(args.model_name)

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = args.model_name + "_" + args.dataset
    storage_name = "sqlite:///{}.db".format(study_name)

    study = optuna.create_study(direction=args.direction,
                                study_name=study_name,
                                storage=storage_name,
                                load_if_exists=True)
    study.optimize(Objective(args, model_name, X, y), n_trials=args.n_trials)
    print("Best parameters:", study.best_trial.params)

    # Run best trial again and save it!
    model = model_name(study.best_trial.params, args)
    if not args.dataset == 'H_sh_300_options':
        cross_validation(model, X, y, args, save_model=True)
    else:
        training_validation_testing(model, X, y, args)


def main_once(args):
    print("Train model with given hyperparameters")
    X, y ,training_trading_dates, validation_trading_dates, testing_trading_dates= load_data(args)

    model_name = str2model(args.model_name)

    parameters = args.parameters[args.dataset][args.model_name]
    model = model_name(parameters, args)
    if not args.dataset == 'H_sh_300_options':
        sc, time = cross_validation(model, X, y, args)
    else:
        training_validation_testing(model, X, y,training_trading_dates, validation_trading_dates, testing_trading_dates, args)



# python train.py --config config/h_sh_300_options.yml  --model_name LightGBM --n_trials 2 --epochs 30 --log_to_file &
if __name__ == "__main__":
    parser = get_parser()
    arguments = parser.parse_args()
    if arguments.log_to_file:
        logger_conf.init_log(f'{arguments.log_to_file_name}')
    print(arguments)
    if arguments.optimize_hyperparameters:
        main(arguments)
    else:
        # Also load the best parameters
        parser = get_given_parameters_parser()
        arguments = parser.parse_args()
        arguments.gpu_index=7
        main_once(arguments)
