"""TensorFlow utilities and data structures for rapid prototyping.

This module is inspired by:
.. _Structuring Your TensorFlow Models by Danijar Hafner / CC BY-SA 3.0:
   https://danijar.com/structuring-your-tensorflow-models/
.. _Patterns for Fast Prototyping with TensorFlow by Danijar Hafner / CC BY-SA 3.0:
   https://danijar.com/patterns-for-fast-prototyping-with-tensorflow/
"""

__license__ = "MIT License"
__copyright__ = "Copyright (c) 2019 Piotr Januszewski"


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
