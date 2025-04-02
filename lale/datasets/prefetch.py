# Copyright 2025 IBM Corporation
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

import sklearn.datasets

import lale.lib.aif360.datasets
from lale.datasets.movie_review import load_movie_review
from lale.datasets.multitable.fetch_datasets import fetch_go_sales_dataset
from lale.datasets.openml.openml_datasets import download_if_missing
from lale.datasets.uci.uci_datasets import (
    fetch_drugslib,
    fetch_household_power_consumption,
)

openml_experiments = [
    "credit-g",
    "breast-cancer",
    "adult",
    "bank-marketing",
    "Default-of-Credit-Card-Clients-Dataset",
    "heart-disease",
    "law-school-admission-bianry",
    "national-longitudinal-survey-binary",
    "UCI-student-performance-mat",
    "UCI-student-performance-por",
    "tae",
    "us_crime",
    "ricci",
    "SpeedDating",
    "nursery",
    "titanic",
    "cloud",
]


def fetch_fairness_dbs():
    dataset_names = {
        "adult": "adult",
        "bank": "bank",
        "compas": "compas",
        "compas_violent": "compas_violent",
        "creditg": "creditg",
        "default_credit": "default_credit",
        "heart_disease": "heart_disease",
        "law_school": "law_school",
        # "meps19": "meps_panel19_fy2015",
        # "meps20": "meps_panel20_fy2015",
        # "meps21": "meps_panel21_fy2016",
        "nlsy": "nlsy",
        "nursery": "nursery",
        "ricci": "ricci",
        "speeddating": "speeddating",
        "student_math": "student_math",
        "student_por": "student_por",
        "tae": "tae",
        "titanic": "titanic",
        "us_crime": "us_crime",
    }

    def try_fetch(dataset_name):
        long_name = dataset_names[dataset_name]
        fetcher_function = getattr(lale.lib.aif360.datasets, f"fetch_{long_name}_df")
        try:
            X, y, fairness_info = fetcher_function()
        except SystemExit:
            print(f"skipping {dataset_name} because it is not downloaded")
            return None
        return X, y, fairness_info

    for name in dataset_names:
        try_fetch(name)


def prefetch_data():
    load_movie_review()

    fetch_go_sales_dataset()

    fetch_drugslib()
    fetch_household_power_consumption()

    for name in openml_experiments:
        download_if_missing(name, True)

    fetch_fairness_dbs()

    sklearn.datasets.fetch_california_housing()
    sklearn.datasets.load_digits()
    sklearn.datasets.load_iris()
    sklearn.datasets.fetch_20newsgroups()
    sklearn.datasets.load_diabetes()
    sklearn.datasets.fetch_covtype()
    sklearn.datasets.load_diabetes()
    sklearn.datasets.fetch_openml(name="house_prices", as_frame=True)
    sklearn.datasets.load_breast_cancer()


def main():
    prefetch_data()


if __name__ == "__main__":
    main()
