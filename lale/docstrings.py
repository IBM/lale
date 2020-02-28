import lale.helpers
import lale.pretty_print

def add_indent(string, amount):
    lines = string.splitlines()
    indent = ' ' * amount
    indented = [indent + line for line in lines]
    result = '\n'.join(indented)
    return result

def _params_docstring(params_schema):
    if len(params_schema['properties']) == 0:
        result = ''
    else:
        result = 'Parameters\n----------\n'
    for param_name, param_schema in params_schema['properties'].items():
        result += param_name + ': '
        tags = []
        if 'relevantToOptimizer' in params_schema:
            if param_name in params_schema['relevantToOptimizer']:
                tags.append('relevant to optimizer')
            else:
                tags.append('not tuned by optimizer')
        if 'required' in params_schema:
            if param_name not in params_schema['required']:
                tags.append('optional')
        result += ', '.join(tags)
        result += '\n'
        result += add_indent(param_schema['description'], 2)
        result += '\n\n  .. code:: python\n\n'
        schema = lale.helpers.dict_without(param_schema, 'description')
        schema_string = lale.pretty_print.schema_to_string(schema)
        result += add_indent('schema = ' + schema_string, 4)
        result += '\n\n'
    return result

def _method_docstring(description, params_schema, result_schema=None):
    result = description + '\n\n'
    result += _params_docstring(params_schema)
    if result_schema is not None:
        result += 'Returns\n-------\nSee schema.\n'
        result += add_indent(result_schema['description'], 2)
        result += '\n\n  .. code:: python\n\n'
        schema = lale.helpers.dict_without(result_schema, 'description')
        schema_string = lale.pretty_print.schema_to_string(schema)
        result += add_indent('schema = ' + schema_string, 4)
    return result

def _hyperparams_docstring(hyperparams_schema):
    result = _params_docstring(hyperparams_schema['allOf'][0])
    assert len(hyperparams_schema['allOf']) == 1, 'not yet implemented'
    return result

def _cls_docstring(impl_cls, combined_schemas):
    result = combined_schemas['description']
    module_name = impl_cls.__module__
    cls_name = impl_cls.__name__
    result += f'\n\nInstead of using `{module_name}.{cls_name}` directly,\nuse its wrapper, `{module_name[:module_name.rfind(".")]}.{cls_name[:-4]}`.\n\n'
    hyperparams_schema = combined_schemas['properties']['hyperparams']
    result += _hyperparams_docstring(hyperparams_schema)
    return result

def set_docstrings(impl_cls, combined_schemas):
    assert impl_cls.__doc__ is None
    impl_cls.__doc__ = _cls_docstring(impl_cls, combined_schemas)
    if hasattr(impl_cls, 'fit'):
        assert impl_cls.fit.__doc__ is None
        impl_cls.fit.__doc__ = _method_docstring(
            'Train the operator.',
            combined_schemas['properties']['input_fit'])
    if hasattr(impl_cls, 'transform'):
        assert impl_cls.transform.__doc__ is None
        impl_cls.transform.__doc__ = _method_docstring(
            'Transform the data.',
            combined_schemas['properties']['input_predict'],
            combined_schemas['properties']['output_transform'])
