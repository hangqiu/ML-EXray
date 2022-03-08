import os

import numpy as np

from sklearn import preprocessing


class Assertions(object):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def validate_inputs(self, model_value, ref_model_value):
        assert model_value.shape == ref_model_value.shape, f"Model input {model_value.shape} and Reference {ref_model_value.shape} are not the same shape"


class ChannelAssertions(Assertions):
    def __call__(self, model_value, ref_model_value, permute_order, *args, **kwargs):
        """
        Permuting the channel of model input, check if all items are equal to ref_model_value

        :param model_value: numpy array
        :param ref_model_value: numpy array
        :param permute_order: list of channel
        :param args:
        :param kwargs:
        :return:
        """

        #TODO: make permute_order None by default and check all permutations
        if np.allclose(model_value,ref_model_value):
            return
        self.validate_inputs(model_value, ref_model_value)

        assert len(permute_order) == len(model_value.shape), f"Permute order {permute_order} length {len(permute_order)} not matching input shape {model_value.shape} length {len(model_value.shape)}"

        permuted_model_value = np.transpose(model_value, permute_order)
        if np.allclose(permuted_model_value, ref_model_value):
            raise AssertionError(f'Channel arrangement mismatch, need to permute as {permute_order}')



class OrientationAssertions(Assertions):
    valid_rotations = [-270, -180, -90, 90, 180, 270]

    def validate_inputs(self, model_value, ref_model_value):
        super().validate_inputs(model_value,ref_model_value)
        assert len(model_value.shape) >= 2, f"Input Dimension {model_value.shape} is less than 2, invalid for Orientation Assertion"

    def __call__(self, model_value, ref_model_value, rotation, *args, **kwargs):
        """
        Rotate the model input, check if all items are equal to ref_model_value

        :param model_value:
        :param ref_model_value:
        :param rotation: [None, 90, -270, 180, -180, 270, -90]
        :param args:
        :param kwargs:
        :return:
        """
        if (rotation is not None) and (rotation not in self.valid_rotations):
            raise ValueError(f"rotation {rotation} is not in valid options {self.valid_rotations}")

        if np.allclose(model_value, ref_model_value):
            return
        self.validate_inputs(model_value, ref_model_value)

        if rotation is not None and rotation < 0:
            rotation += 360

        model_value_rot90 = np.rot90(model_value)
        model_value_rot180 = np.rot90(model_value_rot90)
        model_value_rot270 = np.rot90(model_value_rot180)

        if rotation is None or rotation==90:
            if np.allclose(model_value_rot90, ref_model_value):
                raise AssertionError(f'Orientation mismatch, need to rotate 90 degree counter-clockwise')

        if rotation is None or rotation == 180:
            if np.allclose(model_value_rot180, ref_model_value):
                raise AssertionError(f'Orientation mismatch, need to rotate 180 degree')

        if rotation is None or rotation == 270:
            if np.allclose(model_value_rot270, ref_model_value):
                raise AssertionError(f'Orientation mismatch, need to rotate 90 degree clockwise')


class NormalizationAssertions(Assertions):
    def __call__(self, model_value, ref_model_value, *args, **kwargs):
        """
        Re-normalize the model input to [0,1], check if all items are equal to ref_model_value

        :param model_value:
        :param ref_model_value:
        :param args:
        :param kwargs:
        :return:
        """

        if np.allclose(model_value, ref_model_value):
            return
        self.validate_inputs(model_value, ref_model_value)

        normalized_model_value = preprocessing.normalize(model_value)
        normalized_ref_model_value = preprocessing.normalize(ref_model_value)
        if np.allclose(normalized_model_value, normalized_ref_model_value):
            raise AssertionError(f"Normalization mismatch: "
                                 f"model [{np.min(model_value)},{np.max(model_value)}], "
                                 f"reference [{np.min(ref_model_value)},{np.max(ref_model_value)}]")



# unit tests
if __name__ == "__main__":
    mRotAssert = OrientationAssertions()
    mChannelAssert = ChannelAssertions()
    mNormAssert = NormalizationAssertions()
    a = np.array([[1,2],[3,4]])
    # all assertions should pass same input check
    mRotAssert(a,a, rotation=None)
    mChannelAssert(a,a,permute_order=[1,0])
    mNormAssert(a,a)

    # orientation
    try:
        b = np.array([[2,4],[1,3]])
        mRotAssert(a, b, rotation=None)
    except AssertionError as e:
        print(e)

    try:
        c = np.transpose(a,[1,0])
        mChannelAssert(a,c,permute_order=[1,0])
    except AssertionError as e:
        print(e)

    try:
        d = a * 100
        mNormAssert(a, d)
    except AssertionError as e:
        print(e)