from copy import deepcopy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as torch_utils

from ignite.engine import Engine
from ignite.engine import Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from utils import get_grad_norm, get_parameter_norm


VERBOSE_EPOCH_WISE=1
VERBOSE_BATCH_WISE=1

class MyEngine(Engine):

    def __init__(self, func, model, crit, optimizer, config):
        self.model = model
        self.crit = crit
        self.optimizer = optimizer
        self.config = config

        super().__init__(func)

        self.best_loss = np.inf
        self.best_model = None

        self.device = next(model.parameters()).device

    @staticmethod
    def train(engine, mini_batch):
        engine.model.train()
        engine.optimizer.zero_grad()
        x, y = mini_batch
        x, y = x.to(engine.device), y.to(engine.device)

        y_hat = engine.model(x)

        loss = engine.crit(y_hat, y)
        loss.backward()

        # Classification인지 확인
        if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
            accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
        else :
            accuracy = 0

        # parameter L2 Norm, Model parameter 학습될수록 높아짐
        p_norm = float(get_parameter_norm(engine.model.parameters()))
        # Gradient L2 Norm, loss surface의 가파름 정도
        g_norm = float(get_grad_norm(engine.model.parameters()))

        engine.optimizer.step()

        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            '|param|': p_norm,
            '|g_param|': g_norm,
        }

    @staticmethod
    def validate(engine, mini_batch):
        engine.model.eval()

        with torch.no_grad():
            x, y = mini_batch
            x, y = x.to(engine.device), y.to(engine.device)

            y_hat = engine.model(x)

            loss = engine.crit(y_hat, y)

            if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
                accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
            else :
                accuracy = 0

        return {
            'loss': float(loss),
            'accuracy': float(accuracy)
        }

    
    @staticmethod
    def attach(train_engine, validation_engine, verbose=VERBOSE_BATCH_WISE):

        def attach_running_average(engine, metric_name):
            RunningAverage(output_transform=lambda x: x[metric_name]).attach(
                engine,
                metric_name
            )

        training_metric_names = ['loss', 'accuracy', '|param|', '|g_param|']

        for metric_name in training_metric_names:
            attach_running_average(train_engine, metric_name)

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(train_engine, training_metric_names)

        if verbose >= VERBOSE_EPOCH_WISE:
            @train_engine.on(Events.EPOCH_COMPLETED)
            def print_train_logs(engine):
                print('Epoch {} - |param|={:.2e} |g-param|={:.2e} loss={:.4e} accuracy={:.4f}'.format(
                    engine.state.epoch,
                    engine.state.metrics['|param|'],
                    engine.state.metrics['|g_param|'],
                    engine.state.metrics['loss'],
                    engine.state.metrics['accuracy'],
                ))

        validation_metric_names = ['loss', 'accuracy']

        for metric_name in validation_metric_names:
            attach_running_average(validation_engine, metric_name)

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(validation_engine, validation_metric_names)

        if verbose >= VERBOSE_EPOCH_WISE:
            @validation_engine.on(Events.EPOCH_COMPLETED)
            def print_validation_logs(engine):
                print('Validation - loss={:.4e} accuracy={:.4f} best_loss={:.4e}'.format(
                    engine.state.metrics['loss'],
                    engine.state.metrics['accuracy'],
                    engine.best_loss,
                ))

    @staticmethod
    def check_best(engine):
        loss = float(engine.state.metrics['loss'])
        if loss <= engine.best_loss:
            engine.best_loss = loss
            engine.best_model = deepcopy(engine.model.state_dict())

    @staticmethod
    def save_model(engine, train_engine, config, **kwargs):
        torch.save({
            'model': engine.best_model,
            'config': config,
            **kwargs
        }, config.model_fn)



class Trainer_ignite():

    def __init__(self, config):
        self.config = config

    def train(self,
              model, crit, optimizer,
              train_loader , valid_loader):
        train_engine = MyEngine(
            MyEngine.train,
            model, crit, optimizer, self.config
        )
        validation_engine = MyEngine(
            MyEngine.validate,
            model, crit, optimizer, self.config
        )

        MyEngine.attach(
            train_engine,
            validation_engine,
            verbose=self.config.verbose
        )

        def run_validation(engine, validation_engine, valid_loader):
            validation_engine.run(valid_loader, max_epochs=1)

        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            run_validation,
            validation_engine, valid_loader,
        )
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            MyEngine.check_best,
        )
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            MyEngine.save_model,
            train_engine, self.config
        )

        train_engine.run(
            train_loader,
            max_epochs=self.config.n_epochs
        )

        return model



# class Trainer():

#     def __init__(self, model, optimizer, crit):
#         self.model = model
#         self.optimizer = optimizer
#         self.crit = crit

#         super().__init__()


#     def _batchify(self, x, y, batch_size, random_split=True):
#         if random_split :
#             indices = torch.randperm(x.size(0), device=x.device)
#             x = torch.index_select(x, dim=0, index=indices)
#             y = torch.index_select(y, dim=0, index=indices)

#         x = x.split(batch_size, dim=0)
#         y = y.split(batch_size, dim=0)

#         return x, y


#     def _train(self, x, y, config):
#         self.model.train()

#         x, y = self._batchify(x, y, config.batch_size)
#         total_loss = 0

#         for i, (x_i, y_i) in enumerate(zip(x, y)):
#             y_hat_i = self.model(x_i)
#             loss_i = self.crit(y_hat_i, y_i.squeeze())

#             self.optimizer.zero_grad()
#             loss_i.backward()

#             self.optimizer.step()

#             if config.verbose >= 2:
#                 print("Train Iteration(%d/%d): loss=%.4e" % (i + 1, len(x), float(loss_i)))

#             total_loss += float(loss_i)
        
#         return total_loss / len(x)


#     def _validate(self, x, y, config):
#         self.model.eval()

#         x, y = self._batchify(x, y, config.batch_size, random_split=False)
#         total_loss = 0

#         with torch.no_grad():
#             for i, (x_i, y_i) in enumerate(zip(x, y)):
#                 y_hat_i = self.model(x_i)
#                 loss = self.crit(y_hat_i, y_i.squeeze())

#                 if config.verbose >= 2:
#                     print("Valid Iteration(%d/%d): loss=%.4e" % (i + 1, len(x), float(loss)))
#                 total_loss += float(loss)

#         return total_loss / len(x)


#     def train(self, train_data, valid_data, config):
#         lowest_loss = np.inf
#         best_model = None

#         for epoch_index in range(config.n_epochs):
#             train_loss = self._train(train_data[0], train_data[1], config)
#             valid_loss = self._validate(valid_data[0], valid_data[1], config)

#             if valid_loss <= lowest_loss:
#                 lowest_loss = valid_loss
#                 best_model = deepcopy(self.model.state_dict())

#             print("Epoch(%d/%d): train_loss=%.4e  valid_loss=%.4e  lowest_loss=%.4e" % (
#                 epoch_index + 1,
#                 config.n_epochs,
#                 train_loss,
#                 valid_loss,
#                 lowest_loss
#             ))

#         self.model.load_state_dict(best_model)
