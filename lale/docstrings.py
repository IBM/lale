import lale.helpers
import lale.pretty_print
import pprint

def _indent(prefix, string, first_prefix=None):
    lines = string.splitlines()
    if first_prefix is None:
        first_prefix = prefix
    first_indented = (first_prefix + lines[0]).rstrip()
    rest_indented = [(prefix + line).rstrip() for line in lines[1:]]
    result = first_indented + '\n' + '\n'.join(rest_indented)
    return result

def _value_docstring(value):
    return pprint.pformat(value, width=10000, compact=True)

def _kind_tag(schema):
    if 'type' in schema:
        if schema['type'] == 'object':
            return 'dict'
        else:
            return schema['type']
    elif 'enum' in schema:
        values = schema['enum']
        assert len(values) >= 1
        if len(values) == 1:
            return _value_docstring(values[0])
        elif len(values) == 2:
            return ' or '.join([_value_docstring(v) for v in values])
        else:
            prefix = ', '.join([_value_docstring(v) for v in values[:-1]])
            suffix = ', or ' + _value_docstring(values[-1])
            return prefix + suffix
    elif 'anyOf' in schema:
        return 'union type'
    elif 'allOf' in schema:
        return 'intersection type'
    elif 'not' in schema:
        return 'negated type'
    else:
        return 'any type'

def _schema_docstring(name, schema, required=True, relevant=True):
    tags = []
    if 'typeForOptimizer' in schema:
        tags.append(schema['typeForOptimizer'])
    tags.append(_kind_tag(schema))
    if 'minimum' in schema:
        op = '>' if schema.get('exclusiveMinimum', False) else '>='
        tags.append(op + _value_docstring(schema['minimum']))
    if 'minimumForOptimizer' in schema:
        tags.append('>=' + _value_docstring(schema['minimumForOptimizer'])
                    + ' for optimizer')
    if 'maximum' in schema:
        op = '<' if schema.get('exclusiveMaximum', False) else '<='
        tags.append(op + _value_docstring(schema['maximum']))
    if 'maximumForOptimizer' in schema:
        tags.append('<=' + _value_docstring(schema['maximumForOptimizer'])
                    + ' for optimizer')
    #TODO: for numbers, distribution
    #TODO: for arrays, {min,max}Items, {min,max}ItemsForOptimizer
    if not required:
        tags.append('optional')
    if not relevant or schema.get('forOptimizer', False):
        tags.append('not for optimizer')
    if 'default' in schema:
        tags.append('default ' + _value_docstring(schema['default']))
    result = name + ' : ' if name else ''
    result += ', '.join(tags)
    assert len(result) > 0 and result.rstrip() == result
    if 'description' in schema:
        result += '\n\n' + _indent('  ', schema['description']).rstrip()
    def item_docstring(name, item_schema):
        sd = _schema_docstring(name, item_schema)
        return _indent('    ', sd, '  - ').rstrip()
    body = None
    if 'anyOf' in schema:
        item_docstrings = [item_docstring(None, s) for s in schema['anyOf']]
        body = '\n\n'.join(item_docstrings)
    elif 'allOf' in schema:
        item_docstrings = [item_docstring(None, s) for s in schema['allOf']]
        body = '\n\n'.join(item_docstrings)
    elif 'not' in schema:
        body = item_docstring(None, schema['not'])
    elif schema.get('type', '') == 'array':
        items_schemas = schema['items']
        assert isinstance(items_schemas, dict), 'TODO: array items as list'
        body = item_docstring('items', items_schemas)
    elif schema.get('type', '') == 'object' and 'properties' in schema:
        #TODO: pass down info on which properties are required
        item_docstrings = [item_docstring(k, s)
                           for k, s in schema['properties'].items()]
        body = '\n\n'.join(item_docstrings)
    if body is not None:
        result += '\n\n' + body
    return result.rstrip()

def _params_docstring(params_schema):
    if len(params_schema['properties']) == 0:
        result = ''
    else:
        result = 'Parameters\n----------\n'
    for param_name, param_schema in params_schema['properties'].items():
        required = param_name in params_schema.get('required', {})
        relevant = ('relevantToOptimizer' not in params_schema
                    or param_name in params_schema['relevantToOptimizer'])
        item_docstring = _schema_docstring(
            param_name, param_schema, required, relevant)
        result += _indent('  ', item_docstring, '').rstrip()
        result += '\n\n'
    return result

def _method_docstring(description, params_schema, result_schema=None):
    result = description + '\n\n'
    result += _params_docstring(params_schema)
    if result_schema is not None:
        result += 'Returns\n-------\n'
        item_docstring = _schema_docstring('result', result_schema)
        result += _indent('  ', item_docstring, '')
        result += '\n\n'
    return result

def _hyperparams_docstring(hyperparams_schema):
    result = _params_docstring(hyperparams_schema['allOf'][0])
    assert len(hyperparams_schema['allOf']) == 1, 'TODO: constraints'
    return result

def _cls_docstring(impl_cls, combined_schemas):
    descr_lines = combined_schemas['description'].splitlines()
    result = descr_lines[0]
    module_name = impl_cls.__module__
    cls_name = impl_cls.__name__
    result += f'\n\nInstead of using `{module_name}.{cls_name}` directly,\nuse its wrapper, `{module_name[:module_name.rfind(".")]}.{cls_name[:-4]}`.\n\n'
    result += 'This documentation is auto-generated from JSON schemas.\n\n'
    more_description = '\n'.join(descr_lines[1:]).strip()
    if more_description != '':
        result += more_description + '\n\n'
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
    if hasattr(impl_cls, 'predict'):
        assert impl_cls.predict.__doc__ is None
        impl_cls.predict.__doc__ = _method_docstring(
            'Make predictions.',
            combined_schemas['properties']['input_predict'],
            combined_schemas['properties']['output_predict'])
