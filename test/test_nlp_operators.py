import unittest

import lale.type_checking


class TestTextEncoders(unittest.TestCase):
    def setUp(self):

        self.X_train = [
            "Boston locates in the East Coast",
            "Boston Celtics is part of the East conference of NBA",
            "Cambridge is part of the Greater Boston Area",
            "Manhattan is located in the lower part of NYC",
            "People worked at New York city usually lives in New Jersey Area"
            "The financial center in the world is New York",
        ]

        self.y_train = [0, 0, 0, 1, 1, 1]


def create_function_test_encoder(encoder_name):
    def test_encoder(self):
        import importlib

        module_name = ".".join(encoder_name.split(".")[0:-1])
        class_name = encoder_name.split(".")[-1]
        module = importlib.import_module(module_name)

        class_ = getattr(module, class_name)
        encoder = class_()

        # test_schemas_are_schemas
        lale.type_checking.validate_is_schema(encoder.input_schema_fit())
        lale.type_checking.validate_is_schema(encoder.input_schema_transform())
        lale.type_checking.validate_is_schema(encoder.output_schema_transform())
        lale.type_checking.validate_is_schema(encoder.hyperparam_schema())

        # test_init_fit_transform
        trained = encoder.fit(self.X_train, self.y_train)
        _ = trained.transform(self.X_train)

    test_encoder.__name__ = "test_{0}".format(encoder_name.split(".")[-1])
    return test_encoder


encoders = ["lale.lib.tensorflow.USEPretrainedEncoder"]
for encoder in encoders:
    setattr(
        TestTextEncoders,
        "test_{0}".format(encoder.split(".")[-1]),
        create_function_test_encoder(encoder),
    )
