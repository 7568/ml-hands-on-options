from models.basemodel import BaseModel

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from utils.scorer import get_scorer
import numpy as np
from tqdm import tqdm

from utils.io_utils import get_output_path


class BaseModelTorch(BaseModel):

    def __init__(self, params, args):
        super().__init__(params, args)
        self.device = self.get_device(params, args)
        self.gpus = args.gpu_ids if args.use_gpu and torch.cuda.is_available() and args.data_parallel else None
        self.args = args
        self.params = params

    def to_device(self):
        if self.args.data_parallel:
            self.model = nn.DataParallel(self.model, device_ids=self.args.gpu_ids)

        print("On Device:", self.device)
        self.model.to(self.device)

    def get_device(self, params, args):
        if self.args.use_gpu and torch.cuda.is_available():
            if self.args.data_parallel:
                device = "cuda"  # + ''.join(str(i) + ',' for i in self.args.gpu_ids)[:-1]
            else:
                device = 'cuda'
        else:
            device = 'cpu'
        if self.args.use_gpu:
            device = torch.device(f"cuda:{args.gpu_index}" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def fit(self, X, y, X_val=None, y_val=None):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.params["learning_rate"])

        X = torch.tensor(X).float()
        X_val = torch.tensor(X_val).float()

        y = torch.tensor(y)
        y_val = torch.tensor(y_val)

        if self.args.objective == "regression":
            loss_func = nn.MSELoss()
            y = y.float()
            y_val = y_val.float()
        elif self.args.objective == "classification":
            loss_func = nn.CrossEntropyLoss()
        else:
            loss_func = nn.BCEWithLogitsLoss()
            y = y.float()
            y_val = y_val.float()

        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.args.batch_size)

        # val_dataset = TensorDataset(X_val, y_val)
        # val_loader = DataLoader(dataset=val_dataset, batch_size=self.args.val_batch_size)

        # min_val_loss = float("inf")
        # min_val_loss_idx = 0
        max_f1_score=0
        max_f1_score_idx=0

        loss_history = []
        val_loss_history = []

        for epoch in range(self.args.epochs):
            for i, (batch_X, batch_y) in tqdm(enumerate(train_loader, 0), total=len(train_loader)):
                batch_y = batch_y.to(self.device)
                out = self.model(batch_X.to(self.device))

                if self.args.objective == "regression" or self.args.objective == "binary":
                    out = out.squeeze()

                loss = loss_func(out, batch_y)
                loss_history.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Early Stopping
            # val_loss = 0.0
            # val_dim = 0
            # for val_i, (batch_val_X, batch_val_y) in tqdm(enumerate(val_loader),total=len(val_loader)):
            #     batch_val_y = batch_val_y.to(self.device)
            #
            #     out = self.model(batch_val_X.to(self.device))
            #
            #     if self.args.objective == "regression" or self.args.objective == "binary":
            #         out = out.squeeze()
            #
            #     val_loss += loss_func(out, batch_val_y)
            #     val_dim += 1
            #
            # val_loss /= val_dim
            # val_loss_history.append(val_loss.item())
            #
            # print("Epoch %d, Val Loss: %.5f" % (epoch, val_loss))
            #
            # if val_loss < min_val_loss:
            #     min_val_loss = val_loss
            #     min_val_loss_idx = epoch
            #
            #     # Save the currently best model
            #     self.save_model(filename_extension="best", directory="tmp")

            # Early Stopping
            sc = get_scorer(self.args)
            self.predict(X_val)
            f1_score = sc.eval(y_val, self.predictions, self.prediction_probabilities)['F1 score']
            print("Epoch %d, F1 score: %.5f" % (epoch, f1_score))

            if f1_score > max_f1_score:
                max_f1_score = f1_score
                max_f1_score_idx = epoch

                # Save the currently best model
                self.save_model(filename_extension="best", directory="tmp")

            # if min_val_loss_idx + self.args.early_stopping_rounds < epoch:
            #     print("Validation loss has not improved for %d steps!" % self.args.early_stopping_rounds)
            #     print("Early stopping applies.")
            #     break

            if max_f1_score_idx + self.args.early_stopping_rounds < epoch:
                print("Validation loss has not improved for %d steps!" % self.args.early_stopping_rounds)
                print("Early stopping applies.")
                break

        # Load best model
        self.load_model(filename_extension="best", directory="tmp")
        return loss_history, val_loss_history

    def predict(self, X,testing_trading_dates=None):
        if self.args.objective == "regression":
            if not testing_trading_dates is None:
                self.predictions = self.predict_helper(X,testing_trading_dates)
            else:
                self.predictions = self.predict_helper(X)

        else:
            if not testing_trading_dates is None:
                self.predict_proba(X,testing_trading_dates)
            else:
                self.predict_proba(X)
            self.predictions = np.argmax(self.prediction_probabilities, axis=1)

        return self.predictions

    def predict_proba(self, X: np.ndarray,testing_trading_dates=None) -> np.ndarray:
        if not testing_trading_dates is None:
            probas,sorted_y_test = self.predict_helper(X, testing_trading_dates)
            self.y_test = sorted_y_test
        else:
            probas = self.predict_helper(X)

        probas = np.array(probas).squeeze()
        # If binary task returns only probability for the true class, adapt it to return (N x 2)
        if self.args.objective == "binary" and len(probas.shape)<2:
            probas = probas.reshape(-1,1)
            probas = np.concatenate((1 - probas, probas), 1)

        self.prediction_probabilities = probas
        return self.prediction_probabilities

    def predict_helper(self, X,testing_trading_dates=None):
        self.model.eval()

        X = torch.tensor(X).float()
        test_dataset = TensorDataset(X)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.args.val_batch_size, shuffle=False,
                                 num_workers=2)
        predictions = []
        with torch.no_grad():
            for batch_X in tqdm(test_loader,total=len(test_loader)):
                preds = self.model(batch_X[0].to(self.device))

                if self.args.objective == "binary":
                    preds = torch.sigmoid(preds)

                predictions.append(preds.detach().cpu().numpy())
        return np.concatenate(predictions)

    def save_model(self, filename_extension="", directory="models"):
        filename = get_output_path(self.args, directory=directory, filename="m", extension=filename_extension,
                                   file_type="pt")
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename_extension="", directory="models",device=None):
        filename = get_output_path(self.args, directory=directory, filename="m", extension=filename_extension,
                                   file_type="pt")
        print(filename)
        if device is None:
            state_dict = torch.load(filename)
        else:
            state_dict = torch.load(filename, map_location=torch.device(device))

        self.model.load_state_dict(state_dict)

    def get_model_size(self):
        model_size = sum(t.numel() for t in self.model.parameters() if t.requires_grad)
        return model_size

    @classmethod
    def define_trial_parameters(cls, trial, args):
        raise NotImplementedError("This method has to be implemented by the sub class")
