# The SAINT model.
from models.basemodel_torch import BaseModelTorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from torch import einsum
from einops import rearrange
from sklearn.metrics import f1_score, confusion_matrix
from models.saint_3d_lib.models.pretrainmodel import SAINT as SAINTModel
from models.saint_3d_lib.data_openml import DataSetCatCon
from models.saint_3d_lib.augmentations import embed_data_mask, mixup_data, add_noise
from tqdm import tqdm
from torchmetrics.classification import BinaryF1Score

'''
    batch内数据为一天内的数据
'''


class SAINT_3d(BaseModelTorch):

    def __init__(self, params, args):
        super().__init__(params, args)
        if args.cat_idx:
            num_idx = list(set(range(args.num_features)) - set(args.cat_idx))
            # Appending 1 for CLS token, this is later used to generate embeddings.
            # cat_dims = np.append(np.array([1]), np.array(args.cat_dims)).astype(int)
            cat_dims = np.array(args.cat_dims).astype(int)
        else:
            num_idx = list(range(args.num_features))
            cat_dims = np.array([1])

        # Decreasing some hyperparameter to cope with memory issues
        dim = self.params["dim"] if args.num_features < 50 else 8
        self.batch_size = self.args.batch_size if args.num_features < 50 else 64

        # print("Using dim %d and batch size %d" % (dim, self.batch_size))
        self.cat_dims = cat_dims
        self.model = SAINTModel(
            categories=tuple(cat_dims),
            num_continuous=len(num_idx),
            dim=dim,
            dim_out=2,
            depth=self.params["depth"],  # 6
            heads=self.params["heads"],  # 8
            attn_dropout=self.params["dropout"],  # 0.1
            ff_dropout=self.params["dropout"],  # 0.1
            mlp_hidden_mults=(4, 2),
            cont_embeddings="MLP",
            attentiontype="colrow",
            final_mlp_style="sep",
            y_dim=args.num_classes, device=self.device
        )

        if self.args.data_parallel:
            print(f'self.args.data_parallel :{self.args.data_parallel}')
            self.model.transformer = nn.DataParallel(self.model.transformer, device_ids=self.args.gpu_ids)
            self.model.mlpfory = nn.DataParallel(self.model.mlpfory, device_ids=self.args.gpu_ids)

    def fit(self, X, y, X_val=None, y_val=None, training_trading_dates=None, validation_trading_dates=None):

        if self.args.objective == 'binary':
            criterion = nn.BCEWithLogitsLoss().to(self.device)
            # criterion = nn.BCELoss().to(self.device)
        elif self.args.objective == 'classification':
            criterion = nn.CrossEntropyLoss().to(self.device)
        elif self.args.objective == 'binary_f1':
            criterion = BinaryF1Score().to(self.device)
        else:
            criterion = nn.MSELoss().to(self.device)
        # torch_f1_score = BinaryF1Score().to(self.device)
        self.model.to(self.device)
        print(f'self.learning_rate : {self.args.learning_rate}')
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)

        # SAINT wants it like this...
        X = {'data': X, 'mask': np.ones_like(X)}
        y = {'data': y.reshape(-1, 1)}
        # X_val = {'data': X_val, 'mask': np.ones_like(X_val)}
        # y_val = {'data': y_val.reshape(-1, 1)}

        train_ds = DataSetCatCon(X, y, self.args.cat_idx, self.args.objective, trading_dates=training_trading_dates)
        trainloader = DataLoader(train_ds, batch_size=1, num_workers=4)

        # val_ds = DataSetCatCon(X_val, y_val, self.args.cat_idx, self.args.objective,
        #                        trading_dates=validation_trading_dates)
        # valloader = DataLoader(val_ds, batch_size=1, num_workers=1)

        min_val_loss = float("inf")
        min_val_loss_idx = 0
        max_f1 = 0
        loss_history = []
        val_loss_history = []

        for epoch in range(self.args.epochs):
            self.model.train()

            for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):

                # x_categ is the the categorical data,
                # x_cont has continuous data,
                # y_gts has ground truth ys.
                # cat_mask is an array of ones same shape as x_categ and an additional column(corresponding to CLS
                # token) set to 0s.
                # con_mask is an array of ones same shape as x_cont.
                x_categ, x_cont, y_gts, cat_mask, con_mask = data
                x_categ = x_categ.squeeze(0)
                x_cont = x_cont.squeeze(0)
                y_gts = y_gts.squeeze(0)
                cat_mask = cat_mask.squeeze(0)
                con_mask = con_mask.squeeze(0)

                x_categ, x_cont = x_categ.to(self.device), x_cont.to(self.device)
                cat_mask, con_mask = cat_mask.to(self.device), con_mask.to(self.device)

                # We are converting the data to embeddings in the next step
                _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, self.model)

                reps = self.model.transformer(x_categ_enc, x_cont_enc)

                # select only the representations corresponding to CLS token
                # and apply mlp on it in the next step to get the predictions.
                y_reps = reps[:, -1, :]

                y_outs = self.model.mlpfory(y_reps)

                # if self.args.objective == "binary":
                #     soft_max_y_outs = torch.sigmoid(y_outs)
                # elif self.args.objective == "classification":
                #     soft_max_y_outs = F.softmax(y_outs, dim=1)
                # else:
                #     soft_max_y_outs = y_outs

                if self.args.objective == "regression":
                    y_gts = y_gts.to(self.device)
                elif self.args.objective == "classification":
                    y_gts = y_gts.to(self.device).squeeze()
                else:
                    y_gts = y_gts.to(self.device).float()
                loss = criterion(y_outs, y_gts)
                # loss = 0.01 * criterion(soft_max_y_outs, y_gts) + 0.1 * (1 - f1_score(soft_max_y_outs, y_gts))
                # loss = criterion(soft_max_y_outs, y_gts) * (1.5 - torch_f1_score(soft_max_y_outs, y_gts))
                # loss = criterion(soft_max_y_outs, y_gts) *(1 - self._f1_score(y_gts.detach().cpu(), soft_max_y_outs.detach().cpu()))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_history.append(loss.item())

                # print("Loss", loss.item())

            print(f'epoche " {epoch} , average loss : {np.array(loss_history).mean()}')

            f1 = self.predict_helper(X_val,validation_trading_dates,y_val,tag='validation',need_reload_model=False)

            if f1 > max_f1:
                max_f1 = f1
                min_val_loss_idx = epoch

                # Save the currently best model
                self.save_model(filename_extension=f"{self.args.model_name}_{self.args.learning_rate}_best", directory="tmp")

            if min_val_loss_idx + self.args.early_stopping_rounds < epoch:
                print("Validation loss has not improved for %d steps!" % self.args.early_stopping_rounds)
                print("Early stopping applies.")
                break

            self.predict_helper(self.testing_x, self.testing_trading_dates, self.testing_y,need_reload_model=False)

        # self.load_model(filename_extension="best", directory="tmp")
        return loss_history, val_loss_history

    def pretrain(self, X, y, trading_dates=None, use_pretrain_data=False):

        criterion = nn.CrossEntropyLoss().to(self.device)
        self.model.to(self.device)
        optimizer = optim.AdamW(self.model.parameters(), lr=0.0001)

        # SAINT wants it like this...
        # x_data = np.concatenate((X,X_val),axis=0)
        # y_data_1 = y.reshape(-1, 1)
        # y_data_2 = y_val.reshape(-1, 1)
        # y_data = np.concatenate((y_data_1,y_data_2),axis=0)
        x_data = {'data': X, 'mask': np.ones_like(X)}
        y_data = {'data': y}

        train_ds = DataSetCatCon(x_data, y_data, self.args.cat_idx, self.args.objective, trading_dates=trading_dates)
        trainloader = DataLoader(train_ds, batch_size=1, num_workers=8)

        loss_history = []
        pt_aug_dict = {
            'noise_type': self.args.pt_aug,
            'lambda': self.args.pt_aug_lam
        }
        criterion1 = nn.CrossEntropyLoss()
        criterion2 = nn.MSELoss()
        print("Pretraining begins!")
        for epoch in range(self.args.pretrain_epochs):
            self.model.train()

            for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):

                # x_categ is the the categorical data,
                # x_cont has continuous data,
                # y_gts has ground truth ys.
                # cat_mask is an array of ones same shape as x_categ and an additional column(corresponding to CLS
                # token) set to 0s.
                # con_mask is an array of ones same shape as x_cont.
                x_categ, x_cont, y_gts, cat_mask, con_mask = data
                x_categ = x_categ.squeeze(0)
                x_cont = x_cont.squeeze(0)
                y_gts = y_gts.squeeze(0)
                cat_mask = cat_mask.squeeze(0)
                con_mask = con_mask.squeeze(0)

                x_categ, x_cont = x_categ.to(self.device), x_cont.to(self.device)
                cat_mask, con_mask = cat_mask.to(self.device), con_mask.to(self.device)
                if 'cutmix' in self.args.pt_aug:

                    x_categ_corr, x_cont_corr = add_noise(x_categ, x_cont, noise_params=pt_aug_dict)
                    _, x_categ_enc_2, x_cont_enc_2 = embed_data_mask(x_categ_corr, x_cont_corr, cat_mask, con_mask,
                                                                     self.model)
                else:
                    _, x_categ_enc_2, x_cont_enc_2 = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, self.model)

                if 'mixup' in self.args.pt_aug:
                    x_categ_enc_2, x_cont_enc_2 = mixup_data(x_categ_enc_2, x_cont_enc_2, lam=self.args.mixup_lam)
                # We are converting the data to embeddings in the next step
                _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, self.model)

                loss = 0
                if 'contrastive' in self.args.pt_tasks:
                    aug_features_1 = self.model.transformer(x_categ_enc, x_cont_enc)
                    aug_features_2 = self.model.transformer(x_categ_enc_2, x_cont_enc_2)
                    aug_features_1 = (aug_features_1 / aug_features_1.norm(dim=-1, keepdim=True)).flatten(1, 2)
                    aug_features_2 = (aug_features_2 / aug_features_2.norm(dim=-1, keepdim=True)).flatten(1, 2)
                    if self.args.pt_projhead_style == 'diff':
                        aug_features_1 = self.model.pt_mlp(aug_features_1)
                        aug_features_2 = self.model.pt_mlp2(aug_features_2)
                    elif self.args.pt_projhead_style == 'same':
                        aug_features_1 = self.model.pt_mlp(aug_features_1)
                        aug_features_2 = self.model.pt_mlp(aug_features_2)
                    else:
                        print('Not using projection head')
                    logits_per_aug1 = aug_features_1 @ aug_features_2.t() / self.args.nce_temp
                    logits_per_aug2 = aug_features_2 @ aug_features_1.t() / self.args.nce_temp
                    targets = torch.arange(logits_per_aug1.size(0)).to(logits_per_aug1.device)
                    loss_1 = criterion(logits_per_aug1, targets)
                    loss_2 = criterion(logits_per_aug2, targets)
                    loss = self.args.lam0 * (loss_1 + loss_2) / 2
                elif 'contrastive_sim' in self.args.pt_tasks:
                    aug_features_1 = self.model.transformer(x_categ_enc, x_cont_enc)
                    aug_features_2 = self.model.transformer(x_categ_enc_2, x_cont_enc_2)
                    aug_features_1 = (aug_features_1 / aug_features_1.norm(dim=-1, keepdim=True)).flatten(1, 2)
                    aug_features_2 = (aug_features_2 / aug_features_2.norm(dim=-1, keepdim=True)).flatten(1, 2)
                    aug_features_1 = self.model.pt_mlp(aug_features_1)
                    aug_features_2 = self.model.pt_mlp2(aug_features_2)
                    c1 = aug_features_1 @ aug_features_2.t()
                    loss += self.args.lam1 * torch.diagonal(-1 * c1).add_(1).pow_(2).sum()
                if 'denoising' in self.args.pt_tasks:
                    cat_outs, con_outs = self.model(x_categ_enc_2, x_cont_enc_2)
                    con_outs = torch.cat(con_outs, dim=1)
                    _x_cont = x_cont[:, 0:x_cont.shape[1] // 5]
                    l2 = criterion2(con_outs, _x_cont)
                    l1 = 0
                    _x_categ = x_categ[:, 0:x_categ.shape[1] // 5]
                    for j in range(len(self.cat_dims) // 5):
                        l1 += criterion1(cat_outs[j], _x_categ[:, j])
                    loss += self.args.lam2 * l1 + self.args.lam3 * l2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_history.append(loss.item())

                # print("Loss", loss.item())

            if use_pretrain_data:
                self.save_model(filename_extension="pretrain_2", directory="tmp")
            else:
                self.save_model(filename_extension="pretrain", directory="tmp")
            print(np.array(loss_history).mean())

    def set_testing(self, x, y, testing_trading_dates=None):
        self.testing_x = x
        self.testing_y = y.reshape(-1, 1)
        self.testing_trading_dates = testing_trading_dates

    # def _predict_helper(self, val_dataloader=None):
    #     if val_dataloader is None:
    #         X = {'data': self.testing_x, 'mask': np.ones_like(self.testing_x)}
    #         y = {'data': self.testing_y}
    #
    #         val_ds = DataSetCatCon(X, y, self.args.cat_idx, self.args.objective,
    #                                trading_dates=self.testing_trading_dates)
    #         dataloader = DataLoader(val_ds, batch_size=1, num_workers=1)
    #         print('testing_loader')
    #     else:
    #         print('validation_loader')
    #         dataloader = val_dataloader
    #
    #     _y = []
    #     _pred_y = []
    #     self.model.eval()
    #     with torch.no_grad():
    #         for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
    #             # print(i)
    #             x_categ, x_cont, y_gts, cat_mask, con_mask = data
    #             x_categ = x_categ.squeeze(0)
    #             x_cont = x_cont.squeeze(0)
    #             y_gts = y_gts.squeeze(0)
    #             cat_mask = cat_mask.squeeze(0)
    #             con_mask = con_mask.squeeze(0)
    #             x_categ, x_cont = x_categ.to(self.device), x_cont.to(self.device)
    #             cat_mask, con_mask = cat_mask.to(self.device), con_mask.to(self.device)
    #
    #             _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, self.model)
    #
    #             reps = self.model.transformer(x_categ_enc, x_cont_enc)
    #             y_reps = reps[:, -1, :]
    #
    #             y_outs = self.model.mlpfory(y_reps)
    #             if self.args.objective == "binary":
    #                 y_outs = np.array([int(i>0.5) for i in torch.sigmoid(y_outs).detach().cpu().numpy()])
    #             elif self.args.objective == "classification":
    #                 # y_outs = F.softmax(y_outs, dim=1)
    #                 y_outs = torch.argmax(y_outs, dim=1).detach().cpu()
    #
    #             if self.args.objective == "regression":
    #                 y_gts = y_gts.to(self.device)
    #             elif self.args.objective == "classification":
    #                 y_gts = y_gts.to(self.device).squeeze()
    #             else:
    #                 y_gts = y_gts.to(self.device).float()
    #
    #             # val_loss += criterion(y_outs, y_gts)
    #             # val_dim += 1
    #
    #             _y.append(y_gts.detach().cpu())
    #             _pred_y.append(y_outs)
    #
    #     return _y, _pred_y

    def predict_helper(self, X, _trading_dates=None,y = None,tag='testing',need_reload_model=True):
        X = {'data': X, 'mask': np.ones_like(X)}
        if y is None:
            y = {'data': np.ones((X['data'].shape[0], 1))}
        else:
            y = {'data': y.reshape(-1, 1)}
        _ds = DataSetCatCon(X, y, self.args.cat_idx, self.args.objective, trading_dates=_trading_dates)
        dataloader = DataLoader(_ds, batch_size=1, shuffle=False, num_workers=4)
        print(f'need_reload_model : {need_reload_model}')
        if need_reload_model:
            self.load_model(filename_extension=f"{self.args.model_name}_{self.args.learning_rate}_best",
                            directory="tmp")
            self.model.to(self.device)
        self.model.eval()

        predictions = []
        real_testing_y = []
        with torch.no_grad():

            for data in tqdm(dataloader, total=len(dataloader)):
                x_categ, x_cont, y_gts, cat_mask, con_mask = data
                x_categ = x_categ.squeeze(0)
                x_cont = x_cont.squeeze(0)
                cat_mask = cat_mask.squeeze(0)
                con_mask = con_mask.squeeze(0)

                x_categ, x_cont = x_categ.to(self.device), x_cont.to(self.device)
                cat_mask, con_mask = cat_mask.to(self.device), con_mask.to(self.device)

                _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, self.model)
                reps = self.model.transformer(x_categ_enc, x_cont_enc)
                y_reps = reps[:, -1, :]

                y_outs = self.model.mlpfory(y_reps)
                if self.args.objective == "binary" :
                    y_outs = torch.sigmoid(y_outs).detach().cpu().numpy()
                elif self.args.objective == "regression":
                    y_outs = torch.sigmoid(y_outs-0.5).detach().cpu().numpy()
                else:
                    y_outs = y_outs.detach().cpu()

                real_testing_y.append(y_gts.squeeze(0).detach().cpu())
                predictions.append(y_outs)

        f1 = self._f1_score(real_testing_y, predictions).item()

        print(f'f1 in {tag} : {f1}')

        if tag=='validation':
            return f1
        else:
            return np.concatenate(predictions),np.concatenate(real_testing_y)

    def _f1_score(self, y_true, y_prediction):
        y_true = np.concatenate(y_true).reshape((-1,))
        y_prediction = np.concatenate(y_prediction)
        if self.args.objective == "binary":
            y_prediction_ = np.concatenate((1 - y_prediction, y_prediction), 1)
            y_prediction_ = np.argmax(y_prediction_, axis=1)
        elif self.args.objective == "classification":
            y_prediction_ = torch.argmax(y_prediction, dim=1).detach().cpu().numpy()
        else:
            # y_prediction_ = torch.sigmoid(y_prediction).detach().cpu().numpy()
            y_prediction_ = np.concatenate((1 - y_prediction, y_prediction), 1)
            y_prediction_ = np.argmax(y_prediction_, axis=1)
        print(np.array(y_prediction_))
        print(np.array(y_true))
        print(y_prediction_.shape)
        print(y_true.shape)
        print(f'np.all(y_prediction_==0) : {np.all(y_prediction_ == 0)}')
        print(f'np.all(y_prediction_==1) : {np.all(y_prediction_ == 1)}')
        print(f"np.sum(np.array(y_prediction)) : {np.sum(np.array(y_prediction_))}")
        print(f"np.array(y_prediction).size : {np.array(y_prediction_).size}")
        tn, fp, fn, tp = confusion_matrix(y_true, np.array(y_prediction_)).ravel()
        # print('0：不涨 ， 1：涨')
        print('tn, fp, fn, tp', tn, fp, fn, tp)
        f1 = f1_score(y_true, np.array(y_prediction_), average="binary")
        return f1

    def attribute(self, X, y, strategy=""):
        """ Generate feature attributions for the model input.
            Two strategies are supported: default ("") or "diag". The default strategie takes the sum
            over a column of the attention map, while "diag" returns only the diagonal (feature attention to itself)
            of the attention map.
            return array with the same shape as X.
        """
        global my_attention
        # self.load_model(filename_extension="best", directory="tmp")

        X = {'data': X, 'mask': np.ones_like(X)}
        y = {'data': np.ones((X['data'].shape[0], 1))}

        test_ds = DataSetCatCon(X, y, self.args.cat_idx, self.args.objective)
        testloader = DataLoader(test_ds, batch_size=self.args.val_batch_size, shuffle=False, num_workers=4)

        self.model.eval()
        # print(self.model)
        # Apply hook.
        my_attention = torch.zeros(0)

        def sample_attribution(layer, minput, output):
            global my_attention
            # print(minput)
            """ an hook to extract the attention maps. """
            h = layer.heads
            q, k, v = layer.to_qkv(minput[0]).chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
            sim = einsum('b h i d, b h j d -> b h i j', q, k) * layer.scale
            my_attention = sim.softmax(dim=-1)

        # print(type(self.model.transformer.layers[0][0].fn.fn))
        self.model.transformer.layers[0][0].fn.fn.register_forward_hook(sample_attribution)
        attributions = []
        with torch.no_grad():
            print('test2')
            for data in tqdm(testloader, total=len(testloader)):
                x_categ, x_cont, y_gts, cat_mask, con_mask = data
                x_categ = x_categ.squeeze(0)
                x_cont = x_cont.squeeze(0)
                y_gts = y_gts.squeeze(0)
                cat_mask = cat_mask.squeeze(0)
                con_mask = con_mask.squeeze(0)
                x_categ, x_cont = x_categ.to(self.device), x_cont.to(self.device)
                cat_mask, con_mask = cat_mask.to(self.device), con_mask.to(self.device)
                # print(x_categ.shape, x_cont.shape)
                _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, self.model)
                reps = self.model.transformer(x_categ_enc, x_cont_enc)
                # y_reps = reps[:, 0, :]
                # y_outs = self.model.mlpfory(y_reps)
                if strategy == "diag":
                    attributions.append(my_attention.sum(dim=1)[:, 1:, 1:].diagonal(0, 1, 2))
                else:
                    attributions.append(my_attention.sum(dim=1)[:, 1:, 1:].sum(dim=1))

        attributions = np.concatenate(attributions)
        return attributions

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "dim": trial.suggest_categorical("dim", [32, 64, 128, 256]),
            "depth": trial.suggest_categorical("depth", [1, 2, 3, 6, 12]),
            "heads": trial.suggest_categorical("heads", [2, 4, 8]),
            "dropout": trial.suggest_categorical("dropout", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
        }
        return params
