import unittest

class TestTextEncoders(unittest.TestCase):

    def setUp(self):
        from sklearn.datasets import fetch_20newsgroups
        categories = [
            'alt.atheism',
            'talk.religion.misc',
        ]        
        data = fetch_20newsgroups(subset='train', categories=categories)
        self.X_train, self.y_train = data.data[0:10], data.target[0:10]

def create_function_test_encoder(encoder_name):
    def test_encoder(self):
        X_train, y_train = self.X_train, self.y_train
        import importlib
        module_name = ".".join(encoder_name.split('.')[0:-1])
        class_name = encoder_name.split('.')[-1]
        module = importlib.import_module(module_name)

        class_ = getattr(module, class_name)
        encoder = class_()

        #test_schemas_are_schemas
        from lale.helpers import validate_is_schema
        validate_is_schema(encoder.input_schema_fit())
        validate_is_schema(encoder.input_schema_predict())
        validate_is_schema(encoder.output_schema())
        validate_is_schema(encoder.hyperparam_schema())

        #test_init_fit_transform
        trained = encoder.fit(self.X_train, self.y_train)
        transformed = trained.transform(self.X_train)

    test_encoder.__name__ = 'test_{0}'.format(encoder_name.split('.')[-1])
    return test_encoder

encoders = ['lale.lib.tensorflow.USEPretrainedEncoder']
for encoder in encoders:
    setattr(
        TestTextEncoders,
        'test_{0}'.format(encoder.split('.')[-1]),
        create_function_test_encoder(encoder)
    )
