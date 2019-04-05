"""Tests TensorFlow utilities and data structures for rapid prototyping."""

import tf_utils as tfu
import pytest


@pytest.fixture
def simple_dict():
    return {
        'one': 1,
        'two': 2,
        'three': 3,
        'four': 4,
        'five': 5
    }


@pytest.fixture
def nested_dict():
    return {
        'first': {
            'one': 1,
            'two': 2
        },
        'second': {
            'three': 3,
            'third': {
                'four': 4
            }
        },
        'five': 5
    }


class TestAttrDict(object):

    def test_empty(self):
        ad = tfu.AttrDict()

        assert ad == {}

    def test_simple(self, simple_dict):
        ad = tfu.AttrDict(simple_dict)

        assert ad == simple_dict
        assert ad.one == 1
        assert ad.two == 2
        assert ad.three == 3
        assert ad.four == 4
        assert ad.five == 5

    def test_nested(self, nested_dict):
        ad = tfu.AttrDict(nested_dict)

        assert ad == nested_dict
        assert ad.first.one == 1
        assert ad.first.two == 2
        assert ad.second.three == 3
        assert ad.second.third.four == 4
        assert ad.five == 5

    def test_nested_assign(self, nested_dict):
        ad = tfu.AttrDict(nested_dict)

        ad.first.one = -1
        ad.second.third.four = -4
        ad.five = -5

        assert ad != nested_dict
        assert ad.first.one == -1
        assert ad.first.two == 2
        assert ad.second.three == 3
        assert ad.second.third.four == -4
        assert ad.five == -5

    def test_nested_update(self, nested_dict):
        ad = tfu.AttrDict(nested_dict)

        ad.nested_update({
            'first': {
                'one': -1
            },
            'second': {
                'third': {
                    'four': -4
                }
            },
            'five': -5
        })

        assert ad != nested_dict
        assert ad.first.one == -1
        assert ad.first.two == 2
        assert ad.second.three == 3
        assert ad.second.third.four == -4
        assert ad.five == -5


@pytest.fixture
def simple_json():
    return '''{
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5
    }'''


@pytest.fixture
def nested_json():
    return '''{
        "first": {
            "one": 1,
            "two": 2
        },
        "second": {
            "three": 3,
            "third": {
                "four": 4
            }
        },
        "five": 5
    }'''


class TestAttrDictFromJSON(object):

    def test_empty(self):
        ad = tfu.attrdict_from_json('{}')

        assert ad == {}

    def test_simple(self, simple_dict, simple_json):
        ad = tfu.attrdict_from_json(simple_json)

        assert ad == simple_dict
        assert ad.one == 1
        assert ad.two == 2
        assert ad.three == 3
        assert ad.four == 4
        assert ad.five == 5

    def test_nested(self, nested_dict, nested_json):
        ad = tfu.attrdict_from_json(nested_json)

        assert ad == nested_dict
        assert ad.first.one == 1
        assert ad.first.two == 2
        assert ad.second.three == 3
        assert ad.second.third.four == 4
        assert ad.five == 5


@pytest.fixture
def simple_yaml():
    return '''
        one: 1
        two: 2
        three: 3
        four: 4
        five: 5
    '''


@pytest.fixture
def nested_yaml():
    return '''
        first:
            one: 1
            two: 2
        second:
            three: 3
            third:
                four: 4
        five: 5
    '''


class TestAttrDictFromYAML(object):

    def test_empty(self):
        ad = tfu.attrdict_from_yaml('')

        assert ad == {}

    def test_simple(self, simple_dict, simple_yaml):
        ad = tfu.attrdict_from_yaml(simple_yaml)

        assert ad == simple_dict
        assert ad.one == 1
        assert ad.two == 2
        assert ad.three == 3
        assert ad.four == 4
        assert ad.five == 5

    def test_nested(self, nested_dict, nested_yaml):
        ad = tfu.attrdict_from_yaml(nested_yaml)

        assert ad == nested_dict
        assert ad.first.one == 1
        assert ad.first.two == 2
        assert ad.second.three == 3
        assert ad.second.third.four == 4
        assert ad.five == 5
