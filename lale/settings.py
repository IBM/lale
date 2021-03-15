disable_hyperparams_schema_validation = False
disable_data_schema_validation = True


def set_disable_data_schema_validation(flag: bool):
    """Lale can validate the input and output data used for fit, predict, predict_proba etc.
    against the data schemas defined for an operator. This method allows users to control
    whether the data schema validation should be turned on or not.

        Parameters
        ----------
        flag : bool
            A value of True will disable the data schema validation, and a value of False will enable it.
            It is True by default.
    """
    global disable_data_schema_validation
    disable_data_schema_validation = flag


def set_disable_hyperparams_schema_validation(flag: bool):
    """Lale can validate the hyperparameter values passed while creating an operator against
    the json schema defined for hyperparameters of an operator. This method allows users to control
    whether such validation should be turned on or not.

        Parameters
        ----------
        flag : bool
            A value of True will disable the hyperparameter schema validation, and a value of False will enable it.
            It is False by default.
    """
    global disable_hyperparams_schema_validation
    disable_hyperparams_schema_validation = flag
