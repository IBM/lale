import pprint


def _indent(prefix, string, first_prefix=None):
    lines = string.splitlines()
    if lines:
        if first_prefix is None:
            first_prefix = prefix
        first_indented = (first_prefix + lines[0]).rstrip()
        rest_indented = [(prefix + line).rstrip() for line in lines[1:]]
        result = first_indented + "\n" + "\n".join(rest_indented)
        return result
    else:
        return ""


def _value_docstring(value):
    return pprint.pformat(value, width=10000, compact=True)


def _kind_tag(schema):
    if "anyOf" in schema:
        return "union type"
    elif "allOf" in schema:
        return "intersection type"
    elif "not" in schema or "laleNot" in schema:
        return "negated type"
    elif "type" in schema:
        if schema["type"] == "object":
            return "dict"
        elif schema["type"] == "number":
            return "float"
        elif isinstance(schema["type"], list):
            return " or ".join(schema["type"])
        else:
            return schema["type"]
    elif "enum" in schema:
        values = schema["enum"]
        assert len(values) >= 1
        if len(values) == 1:
            return _value_docstring(values[0])
        elif len(values) == 2:
            return " or ".join([_value_docstring(v) for v in values])
        else:
            prefix = ", ".join([_value_docstring(v) for v in values[:-1]])
            suffix = ", or " + _value_docstring(values[-1])
            return prefix + suffix
    else:
        return "any type"


def _schema_docstring(name, schema, required=True, relevant=True):
    tags = []
    if "laleType" in schema:
        tags.append(schema["laleType"])
    else:
        tags.append(_kind_tag(schema))
    if "minimum" in schema:
        op = ">" if schema.get("exclusiveMinimum", False) else ">="
        tags.append(op + _value_docstring(schema["minimum"]))
    if "minimumForOptimizer" in schema:
        tags.append(
            ">=" + _value_docstring(schema["minimumForOptimizer"]) + " for optimizer"
        )
    if "maximum" in schema:
        op = "<" if schema.get("exclusiveMaximum", False) else "<="
        tags.append(op + _value_docstring(schema["maximum"]))
    if "laleMaximum" in schema:
        tags.append("<=" + _value_docstring(schema["laleMaximum"]))
    if "maximumForOptimizer" in schema:
        tags.append(
            "<=" + _value_docstring(schema["maximumForOptimizer"]) + " for optimizer"
        )
    if "distribution" in schema:
        tags.append(schema["distribution"] + " distribution")
    if "minItems" in schema:
        tags.append(">=" + _value_docstring(schema["minItems"]) + " items")
    if "minItemsForOptimizer" in schema:
        tags.append(
            ">="
            + _value_docstring(schema["minItemsForOptimizer"])
            + " items for optimizer"
        )
    if "maxItems" in schema:
        tags.append("<=" + _value_docstring(schema["maxItems"]) + " items")
    if "maxItemsForOptimizer" in schema:
        tags.append(
            "<="
            + _value_docstring(schema["maxItemsForOptimizer"])
            + " items for optimizer"
        )
    if not required:
        tags.append("optional")
    if not relevant or schema.get("forOptimizer", True) is False:
        tags.append("not for optimizer")
    if "default" in schema:
        tags.append("default " + _value_docstring(schema["default"]))

    def item_docstring(name, item_schema, required=True):
        sd = _schema_docstring(name, item_schema, required=required)
        return _indent("    ", sd, "  - ").rstrip()

    body = None
    if "anyOf" in schema:
        item_docstrings = [item_docstring(None, s) for s in schema["anyOf"]]
        body = "\n\n".join(item_docstrings)
    elif "allOf" in schema:
        item_docstrings = [item_docstring(None, s) for s in schema["allOf"]]
        body = "\n\n".join(item_docstrings)
    elif "not" in schema:
        body = item_docstring(None, schema["not"])
    elif "laleNot" in schema:
        body = f"  - '{schema['laleNot']}'"
    elif schema.get("type", "") == "array":
        if "items" in schema:
            items_schemas = schema["items"]
            if isinstance(items_schemas, dict):
                body = item_docstring("items", items_schemas)
            else:
                items_docstrings = [
                    item_docstring(f"item {i}", s) for i, s in enumerate(items_schemas)
                ]
                body = "\n\n".join(items_docstrings)
    elif schema.get("type", "") == "object" and "properties" in schema:
        item_docstrings = [
            item_docstring(k, s) for k, s in schema["properties"].items()
        ]
        body = "\n\n".join(item_docstrings)
    result = name + " : " if name else ""
    try:
        result += ", ".join(tags)
    except BaseException as e:
        raise ValueError(f"Unexpected internal error for {schema}.") from e
    assert len(result) > 0 and result.rstrip() == result
    if result.startswith("-"):
        result = "\\" + result
    if body is not None and body.find("\n") == -1:
        assert body.startswith("  - ")
        result += " **of** " + body[4:]
    if "description" in schema:
        result += "\n\n" + _indent("  ", schema["description"]).rstrip()
    if body is not None and body.find("\n") != -1:
        result += "\n\n" + body
    return result.rstrip()


def _params_docstring(params_schema):
    params = params_schema.get("properties", {})
    if len(params) == 0:
        result = ""
    else:
        result = "Parameters\n----------\n"
    for param_name, param_schema in params.items():
        required = param_name in params_schema.get("required", {})
        relevant = (
            "relevantToOptimizer" not in params_schema
            or param_name in params_schema["relevantToOptimizer"]
        )
        item_docstring = _schema_docstring(param_name, param_schema, required, relevant)
        result += _indent("  ", item_docstring, "").rstrip()
        result += "\n\n"
    return result


def _method_docstring(description, params_schema, result_schema=None):
    result = description + "\n\n"
    result += _params_docstring(params_schema)
    if result_schema is not None:
        result += "Returns\n-------\n"
        item_docstring = _schema_docstring("result", result_schema)
        result += _indent("  ", item_docstring, "")
        result += "\n\n"
    return result


def _hyperparams_docstring(hyperparams_schema):
    result = _params_docstring(hyperparams_schema["allOf"][0])
    if len(hyperparams_schema["allOf"]) > 1:
        result += "Notes\n-----\n"
        item_docstrings = [
            _schema_docstring(f"constraint {i}", hyperparams_schema["allOf"][i])
            for i in range(1, len(hyperparams_schema["allOf"]))
        ]
        result += "\n\n".join(item_docstrings)
    return result


def _cls_docstring(impl_cls, combined_schemas):
    descr_lines = combined_schemas["description"].splitlines()
    result = descr_lines[0]
    module_name = impl_cls.__module__
    cls_name = impl_cls.__name__
    result += f'\n\nInstead of using `{module_name}.{cls_name}` directly,\nuse its wrapper, `{module_name[:module_name.rfind(".")]}.{cls_name[:-4]}`.\n\n'
    result += "This documentation is auto-generated from JSON schemas.\n\n"
    more_description = "\n".join(descr_lines[1:]).strip()
    if more_description != "":
        result += more_description + "\n\n"
    hyperparams_schema = combined_schemas["properties"]["hyperparams"]
    result += _hyperparams_docstring(hyperparams_schema)
    return result


def set_docstrings(impl_cls, combined_schemas):
    assert impl_cls.__doc__ is None
    impl_cls.__doc__ = _cls_docstring(impl_cls, combined_schemas)
    if hasattr(impl_cls, "fit"):
        fit_doc = _method_docstring(
            "Train the operator.", combined_schemas["properties"]["input_fit"]
        )
        assert impl_cls.fit.__doc__ in [None, fit_doc]
        impl_cls.fit.__doc__ = fit_doc
    if hasattr(impl_cls, "transform"):
        transform_doc = _method_docstring(
            "Transform the data.",
            combined_schemas["properties"]["input_transform"],
            combined_schemas["properties"]["output_transform"],
        )
        assert impl_cls.transform.__doc__ in [None, transform_doc]
        impl_cls.transform.__doc__ = transform_doc
    if hasattr(impl_cls, "predict"):
        predict_doc = _method_docstring(
            "Make predictions.",
            combined_schemas["properties"]["input_predict"],
            combined_schemas["properties"]["output_predict"],
        )
        assert impl_cls.predict.__doc__ in [None, predict_doc]
        impl_cls.predict.__doc__ = predict_doc
    if hasattr(impl_cls, "predict_proba"):
        predict_proba_doc = _method_docstring(
            "Probability estimates for all classes.",
            combined_schemas["properties"]["input_predict_proba"],
            combined_schemas["properties"]["output_predict_proba"],
        )
        assert impl_cls.predict_proba.__doc__ in [None, predict_proba_doc]
        impl_cls.predict_proba.__doc__ = predict_proba_doc
    if hasattr(impl_cls, "decision_function"):
        decision_function_doc = _method_docstring(
            "Confidence scores for all classes.",
            combined_schemas["properties"]["input_decision_function"],
            combined_schemas["properties"]["output_decision_function"],
        )
        assert impl_cls.decision_function.__doc__ in [None, decision_function_doc]
        impl_cls.decision_function.__doc__ = decision_function_doc
