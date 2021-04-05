import inspect
import pprint
from typing import TYPE_CHECKING

import lale.helpers

if TYPE_CHECKING:
    from lale.operators import IndividualOp


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
    if schema.get("transient", False):
        tags.append("transient")
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
    if params_schema is None:
        return ""
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


def _arg_docstring(val):
    if val is None:
        return str("None")
    if isinstance(val, (int, float)):
        return str(val)
    elif isinstance(val, list):
        return [_arg_docstring(x) for x in val]
    elif isinstance(val, dict):
        return {_arg_docstring(k): _arg_docstring(v) for k, v in val.items()}
    else:
        return f'"{str(val)}"'


def _paramlist_docstring(hyperparams_schema) -> str:
    params = hyperparams_schema.get("allOf", None)
    if params is None:
        return ""
    if isinstance(params, list):
        if not params:
            return ""
        params = params[0]
    if params is None:
        return ""
    params = params.get("properties", {})
    if len(params) == 0:
        return ""
    result = ", *"
    for param_name, param_schema in params.items():
        result += f", {param_name}"
        default = param_schema.get("default", None)
        if "default" in param_schema:
            default = param_schema["default"]
            default_str = _arg_docstring(default)
            if default_str is not None:
                result += f"={default_str}"
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


def _method_docstring(description, ready_string, params_schema, result_schema=None):
    result = description + "\n\n"
    if ready_string is not None:
        result += "*Note: " + ready_string + "*\n\n"
        result += (
            "Once this method is available, it will have the following signature: \n\n"
        )
    result += _params_docstring(params_schema)
    if result_schema is not None:
        result += "Returns\n-------\n"
        item_docstring = _schema_docstring("result", result_schema)
        result += _indent("  ", item_docstring, "")
        result += "\n\n"
    return result


def _cls_docstring(cls, combined_schemas):
    descr_lines = combined_schemas["description"].splitlines()
    result = descr_lines[0]
    result += "\n\nThis documentation is auto-generated from JSON schemas.\n\n"
    more_description = "\n".join(descr_lines[1:]).strip()
    if more_description != "":
        result += more_description + "\n\n"
    return result


def _set_docstrings_helper(cls, lale_op, combined_schemas):
    properties = combined_schemas.get("properties", None)

    assert cls.__doc__ is None
    impl_cls = lale_op.impl_class
    cls.__doc__ = _cls_docstring(impl_cls, combined_schemas)

    if properties is not None:
        hyperparams_schema = properties.get("hyperparams", None)
        if hyperparams_schema is not None:
            doc = _hyperparams_docstring(hyperparams_schema)
            try:
                args = _paramlist_docstring(hyperparams_schema)

                code = f"""
def __init__(self{args}):
    pass
"""
                import math

                d = {}
                exec(code, {"nan": math.nan, "inf": math.inf}, d)
                __init__ = d["__init__"]  # type: ignore
            except BaseException as e:
                import warnings

                warnings.warn(
                    f"""While trying to generate a docstring for {cls.__name__}, when trying
to create an init method with the appropriate parameter list, an exception was raised: {e}"""
                )

                def __init__(self):
                    pass

            __init__.__doc__ = doc
            cls.__init__ = __init__

    def make_fun(
        fun_name,
        fake_fun,
        description,
        ready_string,
        params_schema_key,
        result_schema_key=None,
    ):
        params_schema = None
        result_schema = None
        if properties is not None:
            if params_schema_key is not None:
                params_schema = properties.get(params_schema_key, None)
            if result_schema_key is not None:
                result_schema = properties.get(result_schema_key, None)

        if hasattr(impl_cls, fun_name):
            ready_string_to_use = None
            if not hasattr(cls, fun_name):
                ready_string_to_use = ready_string
            doc = _method_docstring(
                description, ready_string_to_use, params_schema, result_schema
            )
            setattr(cls, fun_name, fake_fun)
            fake_fun.__name__ = "fun_name"
            fake_fun.__doc__ = doc

    def fit(self, X, y=None, **fit_params):
        pass

    make_fun(
        "fit",
        fit,
        "Train the operator.",
        "The fit method is not available until this operator is trainable.",
        "input_fit",
    )

    def transform(self, X, y=None):
        pass

    make_fun(
        "transform",
        transform,
        "Transform the data.",
        "The transform method is not available until this operator is trained.",
        "input_transform",
        "output_transform",
    )

    def predict(self, X):
        pass

    make_fun(
        "predict",
        predict,
        "Make predictions.",
        "The predict method is not available until this operator is trained.",
        "input_predict",
        "output_predict",
    )

    def predict_proba(self, X):
        pass

    make_fun(
        "predict_proba",
        predict_proba,
        "Probability estimates for all classes.",
        "The predict_proba method is not available until this operator is trained.",
        "input_predict_proba",
        "output_predict_proba",
    )

    def decision_function(self, X):
        pass

    make_fun(
        "decision_function",
        decision_function,
        "Confidence scores for all classes.",
        "The decision_function method is not available until this operator is trained.",
        "input_decision_function",
        "output_decision_function",
    )


def set_docstrings(lale_op: "IndividualOp"):
    """
    If we are running under sphinx, this will take
    a variable whose value is a lale operator
    and change it to a value of an artificial class
    with appropriately documented methods.
    """
    try:
        if __sphinx_build__:  # type: ignore
            try:

                # impl = lale_op.impl_class
                frm = inspect.stack()[1]
                module = inspect.getmodule(frm[0])
                assert module is not None
                combined_schemas = lale_op._schemas
                name = lale.helpers.arg_name(pos=0, level=1)
                assert name is not None

                # we want to make sure that the Operator constructor args are not shown
                def __init__():
                    pass

                new_class = type(name, (lale_op.__class__,), {"__init__": __init__})
                new_class.__module__ = module.__name__
                module.__dict__[name] = new_class

                _set_docstrings_helper(new_class, lale_op, combined_schemas)
            except NameError as e:
                raise ValueError(e)
    except NameError:
        pass
