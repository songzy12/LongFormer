import unittest

import torch

from longformer_self_attention import LongformerSelfAttention

torch_device = "cuda" if torch.cuda.is_available() else "cpu"


class LongformerModelIntegrationTest(unittest.TestCase):
    def _get_hidden_states(self):
        return torch.tensor(
            [[
                [
                    4.98332758e-01,
                    2.69175139e00,
                    -7.08081422e-03,
                    1.04915401e00,
                    -1.83476661e00,
                    7.67220476e-01,
                    2.98580543e-01,
                    2.84803992e-02,
                ],
                [
                    -7.58357372e-01,
                    4.20635998e-01,
                    -4.04739919e-02,
                    1.59924145e-01,
                    2.05135748e00,
                    -1.15997978e00,
                    5.37166397e-01,
                    2.62873606e-01,
                ],
                [
                    -1.69438001e00,
                    4.17574660e-01,
                    -1.49196962e00,
                    -1.76483717e00,
                    -1.94566312e-01,
                    -1.71183858e00,
                    7.72903565e-01,
                    -1.11557056e00,
                ],
                [
                    5.44028163e-01,
                    2.05466114e-01,
                    -3.63045868e-01,
                    2.41865062e-01,
                    3.20348382e-01,
                    -9.05611176e-01,
                    -1.92690727e-01,
                    -1.19917547e00,
                ],
            ]],
            dtype=torch.float32,
            device=torch_device,
        )

    def test_diagonalize(self):
        hidden_states = self._get_hidden_states()
        hidden_states = hidden_states.reshape(
            (1, 8, 4))  # set seq length = 8, hidden dim = 4
        chunked_hidden_states = LongformerSelfAttention._chunk(
            hidden_states, window_overlap=2)
        window_overlap_size = chunked_hidden_states.shape[2]
        self.assertTrue(window_overlap_size == 4)

        padded_hidden_states = LongformerSelfAttention._pad_and_diagonalize(
            chunked_hidden_states)

        self.assertTrue(
            padded_hidden_states.shape[-1] == chunked_hidden_states.shape[-1] +
            window_overlap_size - 1)

        # first row => [0.4983,  2.6918, -0.0071,  1.0492, 0.0000,  0.0000,  0.0000]
        self.assertTrue(
            torch.allclose(padded_hidden_states[0, 0, 0, :4],
                           chunked_hidden_states[0, 0, 0],
                           atol=1e-3))
        self.assertTrue(
            torch.allclose(
                padded_hidden_states[0, 0, 0, 4:],
                torch.zeros((3, ), device=torch_device, dtype=torch.float32),
                atol=1e-3,
            ))
        # last row => [0.0000,  0.0000,  0.0000, 2.0514, -1.1600,  0.5372,  0.2629]
        self.assertTrue(
            torch.allclose(padded_hidden_states[0, 0, -1, 3:],
                           chunked_hidden_states[0, 0, -1],
                           atol=1e-3))
        self.assertTrue(
            torch.allclose(
                padded_hidden_states[0, 0, -1, :3],
                torch.zeros((3, ), device=torch_device, dtype=torch.float32),
                atol=1e-3,
            ))

    def test_pad_and_transpose_last_two_dims(self):
        hidden_states = self._get_hidden_states()
        self.assertEqual(hidden_states.shape, (1, 4, 8))
        padding = (0, 0, 0, 1)

        padded_hidden_states = LongformerSelfAttention._pad_and_transpose_last_two_dims(
            hidden_states, padding)
        self.assertEqual(padded_hidden_states.shape, (1, 8, 5))

        expected_added_dim = torch.zeros((5, ),
                                         device=torch_device,
                                         dtype=torch.float32)
        self.assertTrue(
            torch.allclose(expected_added_dim,
                           padded_hidden_states[0, -1, :],
                           atol=1e-6))
        self.assertTrue(
            torch.allclose(hidden_states[0, -1, :],
                           padded_hidden_states.view(1, -1)[0, 24:32],
                           atol=1e-6))

    def test_chunk(self):
        hidden_states = self._get_hidden_states()
        batch_size = 1
        seq_length = 8
        hidden_size = 4
        hidden_states = hidden_states.reshape(
            (batch_size, seq_length, hidden_size))

        chunked_hidden_states = LongformerSelfAttention._chunk(
            hidden_states, window_overlap=2)

        # expected slices across chunk and seq length dim
        expected_slice_along_seq_length = torch.tensor(
            [0.4983, -0.7584, -1.6944],
            device=torch_device,
            dtype=torch.float32)
        expected_slice_along_chunk = torch.tensor(
            [0.4983, -1.8348, -0.7584, 2.0514],
            device=torch_device,
            dtype=torch.float32)

        self.assertTrue(
            torch.allclose(chunked_hidden_states[0, :, 0, 0],
                           expected_slice_along_seq_length,
                           atol=1e-3))
        self.assertTrue(
            torch.allclose(chunked_hidden_states[0, 0, :, 0],
                           expected_slice_along_chunk,
                           atol=1e-3))
        self.assertEqual(chunked_hidden_states.shape, (1, 3, 4, 4))

    def test_mask_invalid_locations(self):
        hidden_states = self._get_hidden_states()

        batch_size = 1
        seq_length = 8
        hidden_size = 4
        hidden_states = hidden_states.reshape(
            (batch_size, seq_length, hidden_size))
        chunked_hidden_states = LongformerSelfAttention._chunk(
            hidden_states, window_overlap=2)

        hid_states_1 = chunked_hidden_states.clone()
        LongformerSelfAttention._mask_invalid_locations(hid_states_1, 1)
        self.assertTrue(torch.isinf(hid_states_1).sum().item() == 8)

        hid_states_2 = chunked_hidden_states.clone()
        LongformerSelfAttention._mask_invalid_locations(hid_states_2, 2)
        self.assertTrue(torch.isinf(hid_states_2).sum().item() == 24)

        hid_states_3 = chunked_hidden_states.clone()[:, :, :, :3]
        LongformerSelfAttention._mask_invalid_locations(hid_states_3, 2)
        self.assertTrue(torch.isinf(hid_states_3).sum().item() == 24)

        hid_states_4 = chunked_hidden_states.clone()[:, :, 2:, :]
        LongformerSelfAttention._mask_invalid_locations(hid_states_4, 2)
        self.assertTrue(torch.isinf(hid_states_4).sum().item() == 12)