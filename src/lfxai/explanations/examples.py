import abc
import os
from abc import ABC
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.itertools import product

from LabelFreeXAI.src.lfxai.utils.influence import hessian_vector_product, stack_torch_tensors
from LabelFreeXAI.src.lfxai.utils.load import load_embeddings
from typing import Optional

class ExampleBasedExplainer(ABC):
    def __init__(
        self, model: nn.Module, X_train: torch.Tensor, loss_f: callable, **kwargs
    ):
        self.model = model
        self.X_train = X_train
        self.loss_f = loss_f

    @abc.abstractmethod
    def attribute(
        self, X_test: torch.Tensor, train_idx: list, **kwargs
    ) -> torch.Tensor:
        """

        Args:
            X_test:
            train_idx:
            **kwargs:

        Returns:

        """

    @abc.abstractmethod
    def __str__(self):
        """

        Returns: The name of the method

        """


class InfluenceFunctions(ExampleBasedExplainer, ABC):
    def __init__(
        self,
        model: nn.Module,
        loss_f: callable,
        save_dir: Path,
        X_train: torch.Tensor = None,
    ):
        super().__init__(model, X_train, loss_f)
        self.save_dir = save_dir
        if not save_dir.exists():
            os.makedirs(save_dir)

    def attribute(
        self,
        X_test: torch.Tensor,
        train_idx: list,
        batch_size: int = 1,
        damp: float = 1e-3,
        scale: float = 1000,
        recursion_depth: int = 100,
        **kwargs,
    ) -> torch.Tensor:
        """
        Code adapted from https://github.com/ahmedmalaa/torch-influence-functions/
        This function applies the stochastic estimation approach to evaluating influence function based on the power-series
        approximation of matrix inversion. Recall that the exact inverse Hessian H^-1 can be computed as follows:
        H^-1 = \\sum^\\infty_{i=0} (I - H) ^ j
        This series converges if all the eigen values of H are less than 1.

        Returns:
            return_grads: list of torch tensors, contains product of Hessian and v.
        """

        SUBSAMPLES = batch_size
        NUM_SAMPLES = self.X_train.shape[0]

        loss = [
            self.loss_f(
                self.X_train[idx : idx + 1], self.model(self.X_train[idx : idx + 1])
            )
            for idx in train_idx
        ]

        grads = [
            stack_torch_tensors(
                torch.autograd.grad(
                    loss[k], self.model.encoder.parameters(), create_graph=True
                )
            )
            for k in range(len(train_idx))
        ]

        IHVP_ = [grads[k].detach().cpu().clone() for k in range(len(train_idx))]

        for _ in tqdm(range(recursion_depth), unit="recursion", leave=False):
            sampled_idx = np.random.choice(list(range(NUM_SAMPLES)), SUBSAMPLES)
            sampled_loss = self.loss_f(
                self.X_train[sampled_idx], self.model(self.X_train[sampled_idx])
            )
            IHVP_prev = [IHVP_[k].detach().clone() for k in range(len(train_idx))]
            hvps_ = [
                stack_torch_tensors(
                    hessian_vector_product(sampled_loss, self.model, [IHVP_prev[k]])
                )
                for k in tqdm(range(len(train_idx)), unit="example", leave=False)
            ]
            IHVP_ = [
                g_ + (1 - damp) * ihvp_ - hvp_ / scale
                for (g_, ihvp_, hvp_) in zip(grads, IHVP_prev, hvps_)
            ]

        IHVP_ = [
            IHVP_[k] / (scale * NUM_SAMPLES) for k in range(len(train_idx))
        ]  # Rescale Hessian-Vector products
        IHVP_ = torch.stack(IHVP_, dim=0).reshape(
            (len(train_idx), -1)
        )  # Make a tensor (len(train_idx), n_params)

        test_loss = [
            self.loss_f(X_test[idx : idx + 1], self.model(X_test[idx : idx + 1]))
            for idx in range(len(X_test))
        ]
        test_grads = [
            stack_torch_tensors(
                torch.autograd.grad(
                    test_loss[k], self.model.encoder.parameters(), create_graph=True
                )
            )
            for k in range(len(X_test))
        ]
        test_grads = torch.stack(test_grads, dim=0).reshape((len(X_test), -1))
        attribution = torch.einsum("ab,cb->ac", test_grads, IHVP_)
        return attribution

    def attribute_loader(
        self,
        device: torch.device,
        train_loader: DataLoader,
        test_loader: DataLoader,
        train_loader_replacement: DataLoader,
        recursion_depth: int = 100,
        damp: float = 1e-3,
        scale: float = 1000,
        **kwargs,
    ) -> torch.Tensor:
        attribution = torch.zeros((len(test_loader), len(train_loader)))
        for test_idx, (x_test, _) in enumerate(
            tqdm(test_loader, unit="example", leave=False)
        ):
            x_test = x_test.to(device)
            test_loss = self.loss_f(x_test, self.model(x_test))
            test_grad = (
                stack_torch_tensors(
                    torch.autograd.grad(
                        test_loss, self.model.encoder.parameters(), create_graph=True
                    )
                )
                .detach()
                .clone()
            )
            torch.save(
                test_grad.detach().cpu(), self.save_dir / f"test_grad{test_idx}.pt"
            )
        for train_idx, (x_train, _) in enumerate(
            tqdm(train_loader, unit="example", leave=False)
        ):
            x_train = x_train.to(device)
            loss = self.loss_f(x_train, self.model(x_train))
            grad = stack_torch_tensors(
                torch.autograd.grad(
                    loss, self.model.encoder.parameters(), create_graph=True
                )
            )
            ihvp = grad.detach().clone()
            train_sampler = iter(train_loader_replacement)
            for _ in tqdm(range(recursion_depth), unit="recursion", leave=False):
                X_sample, _ = next(train_sampler)
                X_sample = X_sample.to(device)
                sampled_loss = self.loss_f(X_sample, self.model(X_sample))
                ihvp_prev = ihvp.detach().clone()
                hvp = stack_torch_tensors(
                    hessian_vector_product(sampled_loss, self.model, ihvp_prev)
                )
                ihvp = grad + (1 - damp) * ihvp - hvp / scale
            ihvp = ihvp / (scale * len(train_loader))  # Rescale Hessian-Vector products
            torch.save(ihvp.detach().cpu(), self.save_dir / f"train_ihvp{train_idx}.pt")

        for test_idx, train_idx in product(
            range(len(test_loader)), range(len(train_loader)), leave=False
        ):
            ihvp = torch.load(self.save_dir / f"train_ihvp{train_idx}.pt")
            test_grad = torch.load(self.save_dir / f"test_grad{test_idx}.pt")
            attribution[test_idx, train_idx] = (
                torch.dot(test_grad.flatten(), ihvp.flatten()).detach().clone().cpu()
            )
        return attribution

    def __str__(self):
        return "Influence Functions"


class TracIn(ExampleBasedExplainer, ABC):
    def __init__(
        self,
        model: nn.Module,
        loss_f: callable,
        save_dir: Path,
        X_train: torch.Tensor = None,
    ):
        super().__init__(model, X_train, loss_f)
        self.save_dir = save_dir
        if not save_dir.exists():
            os.makedirs(save_dir)

    def attribute(
        self, X_test: torch.Tensor, train_idx: list, learning_rate: float = 1, **kwargs
    ) -> torch.Tensor:
        attribution = torch.zeros(len(X_test), len(train_idx))
        for checkpoint_file in self.model.checkpoints_files:
            self.model.load_state_dict(torch.load(checkpoint_file), strict=False)
            train_loss = [
                self.loss_f(
                    self.X_train[idx : idx + 1], self.model(self.X_train[idx : idx + 1])
                )
                for idx in train_idx
            ]
            train_grads = [
                stack_torch_tensors(
                    torch.autograd.grad(
                        train_loss[k],
                        self.model.encoder.parameters(),
                        create_graph=True,
                    )
                )
                for k in range(len(train_idx))
            ]
            train_grads = torch.stack(train_grads, dim=0).reshape((len(train_idx), -1))
            test_loss = [
                self.loss_f(X_test[idx : idx + 1], self.model(X_test[idx : idx + 1]))
                for idx in range(len(X_test))
            ]
            test_grads = [
                stack_torch_tensors(
                    torch.autograd.grad(
                        test_loss[k], self.model.encoder.parameters(), create_graph=True
                    )
                )
                for k in range(len(X_test))
            ]
            test_grads = torch.stack(test_grads, dim=0).reshape((len(X_test), -1))
            attribution += torch.einsum("ab,cb->ac", test_grads, train_grads)
        return learning_rate * attribution

    def attribute_loader(
        self,
        device: torch.device,
        train_loader: DataLoader,
        test_loader: DataLoader,
        **kwargs,
    ) -> torch.Tensor:
        attribution = torch.zeros((len(test_loader), len(train_loader)))
        for checkpoint_number, checkpoint_file in enumerate(
            tqdm(self.model.checkpoints_files, unit="checkpoint", leave=False)
        ):
            self.model.load_state_dict(torch.load(checkpoint_file), strict=False)
            for train_idx, (x_train, _) in enumerate(
                tqdm(train_loader, unit="example", leave=False)
            ):
                x_train = x_train.to(device)
                train_loss = self.loss_f(x_train, self.model(x_train))
                train_grad = stack_torch_tensors(
                    torch.autograd.grad(
                        train_loss, self.model.encoder.parameters(), create_graph=True
                    )
                )
                torch.save(
                    train_grad.detach().cpu(),
                    self.save_dir
                    / f"train_checkpoint{checkpoint_number}_grad{train_idx}.pt",
                )
            for test_idx, (x_test, _) in enumerate(
                tqdm(test_loader, unit="example", leave=False)
            ):
                x_test = x_test.to(device)
                test_loss = self.loss_f(x_test, self.model(x_test))
                test_grad = stack_torch_tensors(
                    torch.autograd.grad(
                        test_loss, self.model.encoder.parameters(), create_graph=True
                    )
                )
                torch.save(
                    test_grad.detach().cpu(),
                    self.save_dir
                    / f"test_checkpoint{checkpoint_number}_grad{test_idx}.pt",
                )
            for test_idx, train_idx in product(
                range(len(test_loader)), range(len(train_loader)), leave=False
            ):
                train_grad = torch.load(
                    self.save_dir
                    / f"train_checkpoint{checkpoint_number}_grad{train_idx}.pt"
                )
                test_grad = torch.load(
                    self.save_dir
                    / f"test_checkpoint{checkpoint_number}_grad{test_idx}.pt"
                )
                attribution[test_idx, train_idx] += torch.dot(
                    train_grad.flatten(), test_grad.flatten()
                )
        return attribution

    def __str__(self):
        return "TracIn"


class SimplEx(ExampleBasedExplainer, ABC):
    def __init__(
        self, model: nn.Module, loss_f: callable = None, X_train: torch.Tensor = None
    ):
        super().__init__(model, X_train, loss_f)

    def attribute(
        self,
        X_test: torch.Tensor,
        train_idx: list,
        learning_rate: float = 1,
        batch_size: int = 50,
        **kwargs,
    ) -> torch.Tensor:
        attribution = torch.zeros(len(X_test), len(train_idx))
        train_representations = self.model.encoder(self.X_train[train_idx]).detach()
        n_batch = int(len(X_test) / batch_size)
        for n in tqdm(range(n_batch), unit="batch", leave=False):
            batch_features = X_test[n * batch_size : (n + 1) * batch_size]
            batch_representations = self.model.encoder(batch_features).detach()
            attribution[n * batch_size : (n + 1) * batch_size] = self.compute_weights(
                batch_representations, train_representations
            )
        return attribution

    def attribute_loader(
        self,
        device: torch.device,
        train_loader: DataLoader,
        test_loader: DataLoader,
        batch_size: int = 50,
        embedding_folder: Optional[str] = None,
        label: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        '''
        Added one variable, label, to specify in there is label in the training set. 
        load: can load the embeddings, instead of computing it every time. Should increase the speed
        '''
        H_train = []
        if embedding_folder is None:
            if label:
                for x_train, _ in train_loader:
                    H_train.append(self.model.encoder(x_train.to(device)).detach().cpu())
            else:
                for x_train in train_loader:
                    H_train.append(self.model.encoder(x_train.to(device)).detach().cpu())
                    #try:
                        #self.model.encoder(x_train.to(device))
                    #except Exception as e:
                        #print(x_train, 'encountered an error:', e)
            H_train = torch.cat(H_train)
        else:
            H_train = torch.Tensor(load_embeddings(embedding_folder))
        
        H_test = []
        if label:
            for x_test, _ in test_loader:
                H_test.append(self.model.encoder(x_test.to(device)).detach().cpu())
        else:
            for x_test in test_loader:
                H_test.append(self.model.encoder(x_test.to(device)).detach().cpu())
        H_test = torch.cat(H_test)
        attribution = torch.zeros(len(H_test), len(H_train))
        Errors = []
        n_batch = int(len(H_test) / batch_size)
        #for n in tqdm(range(n_batch), unit="batch", leave=False):
        for n in range(n_batch):
            h_test = H_test[n * batch_size : (n + 1) * batch_size]
            attribution[n * batch_size : (n + 1) * batch_size], errors = self.compute_weights(
                h_test, H_train, **kwargs
            )
            Errors.append(errors)
        return attribution, errors

    @staticmethod
    def compute_weights(
        batch_representations: torch.Tensor,
        train_representations: torch.Tensor,
        n_epoch: int = 35000,
        patience: int = 500, # patience parameter
        min_delta: float = 0.001, # threshold for improvement       
        verbose: bool = False,
    ) -> torch.Tensor:
        '''
        Because the n_epoch parameters was hard to finetune, I implemented a stopping cretarion, that allow to 
        chose a high number of n_epochs, while preventing the algorithm to run for too long if small improvements 
        are observed.
        '''
        preweights = torch.zeros(
            (len(batch_representations), len(train_representations)), requires_grad=True
        )
        optimizer = torch.optim.Adam([preweights])
        errors = []
        best_error = float('inf')
        patience_counter = 0
        norm = ((batch_representations) ** 2).sum() 
        for epoch in range(n_epoch):
            optimizer.zero_grad()
            weights = F.softmax(preweights, dim=-1)
            approx_representations = torch.einsum(
                "ij,jk->ik", weights, train_representations
            )
            error_per_instance = torch.sum((approx_representations - batch_representations) ** 2, dim = 1)
            error = error_per_instance.sum()
            #error = ((approx_representations - batch_representations) ** 2).sum()
            error.backward()
            optimizer.step()
            
            current_error = error.detach()
            #errors.append(current_error)
            errors.append(error_per_instance.detach())

            # Check for improvement
            if (best_error - current_error) > min_delta * norm:
                best_error = current_error
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                if verbose: print(f"Early stopping at epoch {epoch}")
                break
                errors.append(error.detach())
                
        return torch.softmax(preweights, dim=-1).detach(), errors

    def __str__(self):
        return "SimplEx"


class NearestNeighbours(ExampleBasedExplainer, ABC):
    def __init__(
        self, model: nn.Module, loss_f: callable = None, X_train: torch.Tensor = None
    ):
        super().__init__(model, X_train, loss_f)

    def attribute(
        self, X_test: torch.Tensor, train_idx: list, batch_size: int = 500, **kwargs
    ) -> torch.Tensor:
        attribution = torch.zeros(len(X_test), len(train_idx))
        train_representations = (
            self.model.encoder(self.X_train[train_idx]).detach().unsqueeze(0)
        )
        n_batch = int(len(X_test) / batch_size)
        for n in tqdm(range(n_batch), unit="batch", leave=False):
            batch_features = X_test[n * batch_size : (n + 1) * batch_size]
            batch_representations = (
                self.model.encoder(batch_features).detach().unsqueeze(1)
            )
            attribution[n * batch_size : (n + 1) * batch_size] = 1 / torch.sum(
                (batch_representations - train_representations) ** 2, dim=-1
            )
        return attribution

    def attribute_loader(
        self,
        device: torch.device,
        train_loader: DataLoader,
        test_loader: DataLoader,
        batch_size: int = 50,
        **kwargs,
    ) -> torch.Tensor:
        H_train = []
        for x_train, _ in train_loader:
            H_train.append(self.model.encoder(x_train.to(device)).detach().cpu())
        H_train = torch.cat(H_train)
        H_test = []
        for x_test, _ in test_loader:
            H_test.append(self.model.encoder(x_test.to(device)).detach().cpu())
        H_test = torch.cat(H_test)
        attribution = torch.zeros(len(H_test), len(H_train))
        n_batch = int(len(H_test) / batch_size)
        for n in tqdm(range(n_batch), unit="batch", leave=False):
            h_test = H_test[n * batch_size : (n + 1) * batch_size]
            attribution[n * batch_size : (n + 1) * batch_size] = 1 / torch.sum(
                (h_test.unsqueeze(1) - H_train.unsqueeze(0)) ** 2, dim=-1
            )
        return attribution

    def __str__(self):
        return "DKNN"
