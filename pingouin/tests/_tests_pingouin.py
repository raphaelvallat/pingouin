"""Utility function for Pingouin tests."""
import numpy as np

class _TestPingouin(object):
    """Visbrain testing utility methods."""

    OBJ = None
    NEEDED_FILES = {}

    def assert_and_test(self, attr, to_set, to_test='NoAttr'):
        """Assert to obj and test."""
        # Set attribute :
        obj = 'self.OBJ'
        if isinstance(to_set, str):
            exec("{}.{}".format(obj, attr) + "='" + to_set + "'")
        else:
            exec("{}.{}".format(obj, attr) + ' = to_set')
        value = eval("{}.{}".format(obj, attr))
        # Test either to_set or to_test :
        value_to_test = to_set if to_test == 'NoAttr' else to_test
        # Test according to data type :
        if isinstance(value_to_test, np.ndarray):
            # Be sure that arrays have the same shape and dtype :
            value = value.reshape(*value_to_test.shape)
            value = value.astype(value_to_test.dtype)
            np.testing.assert_allclose(value, value_to_test)
        else:
            assert value == value_to_test

    def parent_testing(self, obj, parent):
        """Test setting parent."""
        obj.parent = parent
        assert obj.parent.name == parent.name
