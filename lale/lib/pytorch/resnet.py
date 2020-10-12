# Copyright 2019 IBM Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.models import resnet50

import lale.docstrings
import lale.operators


class ResNet50Impl:
    def __init__(
        self,
        num_classes=10,
        model=None,
        num_epochs=2,
        batch_size=128,
        learning_rate_init=0.1,
        learning_rate="constant",
    ):
        self.num_classes = num_classes
        if model is None:
            # self.model = ResNet(152, num_classes)
            self.model = resnet50(num_classes)
        else:
            self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.learning_rate = learning_rate

    def calculate_learning_rate(self, epoch):
        if self.learning_rate == "constant":
            return self.learning_rate_init
        elif self.learning_rate == "decay":
            optim_factor = 0
            if epoch > 160:
                optim_factor = 3
            elif epoch > 120:
                optim_factor = 2
            elif epoch > 60:
                optim_factor = 1

            return self.learning_rate_init * math.pow(0.2, optim_factor)

    def fit(self, X, y=None):
        trainloader = torch.utils.data.DataLoader(
            X, batch_size=self.batch_size, shuffle=True, num_workers=2
        )
        net = self.model.to(self.device)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for epoch in range(self.num_epochs):
            learning_rate = self.calculate_learning_rate(epoch)
            print(learning_rate)
            optimizer = optim.SGD(
                net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4
            )
            criterion = nn.CrossEntropyLoss()
            print("\n=> Training Epoch %d, LR=%.4f" % (epoch, learning_rate))
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = (
                    inputs.to(self.device),
                    targets.to(self.device),
                )  # GPU settings
                optimizer.zero_grad()
                inputs, targets = Variable(inputs), Variable(targets)
                outputs = net(inputs)  # Forward Propagation
                loss = criterion(outputs, targets)  # Loss
                loss.backward()  # Backward Propagation
                optimizer.step()  # Optimizer update

                train_loss += loss.data.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

                sys.stdout.write("\r")
                sys.stdout.write(
                    "| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%"
                    % (
                        epoch,
                        self.num_epochs,
                        batch_idx + 1,
                        (len(X) // self.batch_size) + 1,
                        loss.data.item(),
                        100.0 * correct / total,
                    )
                )
                sys.stdout.flush()
        return ResNet50Impl(
            self.num_classes,
            self.model,
            self.num_epochs,
            self.batch_size,
            self.learning_rate,
        )

    def predict(self, X):
        net = self.model.to(self.device)
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
        net.eval()
        dataloader = torch.utils.data.DataLoader(
            X, batch_size=self.batch_size, shuffle=False, num_workers=2
        )
        predicted_X = None
        for batch_idx, data in enumerate(dataloader):
            if isinstance(data, list) or isinstance(data, tuple):
                inputs = data[
                    0
                ]  # For standard datasets from torchvision, data is a list with X and y
            inputs = inputs.to(self.device)
            inputs = Variable(inputs, volatile=True)
            outputs = net(inputs)

            _, predicted = torch.max(outputs.data, 1)

            predicted = predicted.detach().cpu().numpy()
            predicted = np.reshape(predicted, (predicted.shape[0], 1))
            if predicted_X is None:
                predicted_X = predicted
            else:
                predicted_X = np.vstack((predicted_X, predicted))
        self.model.train()
        return predicted_X


_input_schema_fit = {
    "description": "Input data schema for training.",
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {"X": {"description": "Pytorch Dataset."}},
}

_input_predict_schema = {
    "description": "Input data schema for predictions.",
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {"X": {"description": "Pytorch Dataset."}},
}

_output_predict_schema = {
    "description": "Output data schema for transformed data.",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}


_hyperparams_schema = {
    "description": "Hyperparameter schema.",
    "allOf": [
        {
            "description": "This first sub-object lists all constructor arguments with their "
            "types, one at a time, omitting cross-argument constraints.",
            "type": "object",
            "additionalProperties": False,
            "required": ["num_classes", "num_epochs", "batch_size", "learning_rate"],
            "relevantToOptimizer": ["num_epochs", "batch_size", "learning_rate"],
            "properties": {
                "num_classes": {
                    "description": "Number of classes.",
                    "type": "integer",
                    "default": 10,
                    "minimum": 2,
                },
                "num_epochs": {
                    "description": "The number of epochs used for training.",
                    "type": "integer",
                    "default": 2,
                    "minimum": 0,
                    "distribution": "uniform",
                    "minimumForOptimizer": 2,
                    "maximumForOptimizer": 200,
                },
                "batch_size": {
                    "description": "Batch size used for training and prediction",
                    "type": "integer",
                    "default": 64,
                    "minimum": 1,
                    "distribution": "uniform",
                    "maximumForOptimizer": 128,
                },
                "learning_rate": {
                    "description": "Learning rate scheme for training.",
                    "enum": ["constant", "decay"],
                    "default": "constant",
                },
                "learning_rate_init": {
                    "description": "Initial value of learning rate for training.",
                    "type": "number",
                    "default": 0.1,
                    "minimum": 0,
                    "distribution": "loguniform",
                    "minimumForOptimizer": 1e-05,
                    "maximumForOptimizer": 0.1,
                },
            },
        }
    ],
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters for a transformer for"
    " pytorch implementation of ResNet50 for image classification.",
    "type": "object",
    "tags": {
        "pre": ["images"],
        "op": ["estimator", "classifier", "~interpretable", "pytorch"],
        "post": [],
    },
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_schema_fit,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
    },
}

lale.docstrings.set_docstrings(ResNet50Impl, _combined_schemas)

ResNet50 = lale.operators.make_operator(ResNet50Impl, _combined_schemas)

if __name__ == "__main__":
    import torchvision.datasets as datasets

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[n / 255.0 for n in [129.3, 124.1, 112.4]],
                std=[n / 255.0 for n in [68.2, 65.4, 70.4]],
            ),
        ]
    )  # meanstd transformation

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[n / 255.0 for n in [129.3, 124.1, 112.4]],
                std=[n / 255.0 for n in [68.2, 65.4, 70.4]],
            ),
        ]
    )

    data_train = datasets.CIFAR10(
        root="/tmp/", download=True, transform=transform_train
    )
    clf = ResNet50(num_classes=10, num_epochs=1)
    fitted_clf = clf.fit(data_train)
    data_test = datasets.CIFAR10(
        root="/tmp/", download=True, train=False, transform=transform_test
    )
    predicted = fitted_clf.predict(data_test)
    from sklearn.metrics import accuracy_score

    print(data_test.test_labels[0:10], predicted[0:10])
    print(accuracy_score(data_test.test_labels, predicted.tolist()))
