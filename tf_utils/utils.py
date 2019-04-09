"""TensorFlow utilities and data structures for rapid prototyping.

This module is inspired by:
.. _Structuring Your TensorFlow Models by Danijar Hafner / CC BY-SA 3.0:
   https://danijar.com/structuring-your-tensorflow-models/
.. _Patterns for Fast Prototyping with TensorFlow by Danijar Hafner / CC BY-SA 3.0:
   https://danijar.com/patterns-for-fast-prototyping-with-tensorflow/
"""

__license__ = "MIT License"
__copyright__ = "Copyright (c) 2019 Piotr Januszewski"

import functools


class AttrDict(dict):
    """Defines an attribute dictionary that allows the user to address keys as
    if they were attributes.

    `AttrDict` inherits from `dict`, so it can be used everywhere where `dict` is expected.
    The __init__ constructs an empty `AttrDict` or recursively initialises an `AttrDict`
    from a (maybe nested) `dict`.

    Args:
        other (dict): Either dictionary (can be nested) to initialise an `AttrDict` from
        or None to construct an empty `AttrDict`.
    """

    def __init__(self, other=None):
        super().__init__()

        if other is not None:
            for key, value in other.items():
                # It is easier to ask for forgiveness, than to ask for permission
                try:
                    self[key] = AttrDict(value)
                except AttributeError:
                    self[key] = value

    def nested_update(self, other):
        """Updates the values in this and the nested `AttrDict`s.

        Args:
            other (dict): Dictionary (can be nested) to update the `AttrDict` from.
        """

        for key, value in other.items():
            # It is easier to ask for forgiveness, than to ask for permission
            try:
                self[key].nested_update(value)
            except AttributeError:
                self[key] = value

        return self

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def attrdict_from_json(data):
    """Constructs an AttrDict from JSON formatted string.

    Args:
        data (str): JSON formatted string.
    """

    from json import loads
    return AttrDict(loads(data))


def attrdict_from_yaml(data):
    """Constructs an AttrDict from YAML formatted string.

    Args:
        data (str): YAML formatted string.
    """

    from yaml import safe_load
    return AttrDict(safe_load(data))


def lazy_property_with_scope(*args, scope_name=None, **kwargs):
    """Property decorator on steroids.

    Property decorator which additionally defines TensorFlow variable scope
    and adds lazy loading.

    This decorator can be used without parentheses if no arguments are provided.

    Args:
        scope_name (str): The scope name. If `None` wrapped function's name is
                          used as the scope name.
        *args: Passed to `tf.variable_scope`.
        **kwargs: Passed to `tf.variable_scope`.
    """

    def define_scope(function, scope_name, *args, **kwargs):
        """Decorator for properties that define TensorFlow operations.

        The wrapped function will only be executed once. Subsequent calls to
        it will directly return the result so that operations are added to
        the graph only once. The operations added by the function live within
        a tf.variable_scope(). If this decorator is used with arguments, they
        will be forwarded to the variable scope. The scope name defaults to
        the name of the wrapped function.

        Args:
            scope_name (str): The scope name. If `None` wrapped function's
                              name is used as the scope name.
            *args: Passed to `tf.variable_scope`.
            **kwargs: Passed to `tf.variable_scope`.
        """

        from tensorflow import variable_scope

        attribute = '_cache_' + function.__name__
        name = scope_name or function.__name__

        @property
        @functools.wraps(function)
        def decorator(self):
            if not hasattr(self, attribute):
                with variable_scope(name, *args, **kwargs):
                    setattr(self, attribute, function(self))
            return getattr(self, attribute)
        return decorator

    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return define_scope(args[0], scope_name=scope_name)
    return lambda function: define_scope(function, scope_name=scope_name,
                                         *args, **kwargs)
