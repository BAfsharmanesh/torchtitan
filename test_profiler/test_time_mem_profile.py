import unittest
import torch

from torchtitan.profilers import (
    TimeProfiler,
    MemoryProfiler,
    ModelProfiler,
    register_timing_hooks,
    SavedActivationContext,
    get_layer_names,
    get_param_act_info,
    save_metrics,
    measure_activation_shape,
    slice_layers_2_fit_gpu,
    get_dummy_input,
)

import time
from collections import OrderedDict


class TestLayerTimeProfiler(unittest.TestCase):
    def setUp(self):
        self.profiler = TimeProfiler(layer_names=["layer1", "layer2"])

    def test_timing_record(self):
        key = "test_layer"
        with self.profiler.record_time(key, sync=False):
            time.sleep(0.1)
        self.assertIn(f"{key}_start", self.profiler.timings)
        self.assertIn(f"{key}_end", self.profiler.timings)
        self.assertGreater(
            self.profiler.timings[f"{key}_end"][0]
            - self.profiler.timings[f"{key}_start"][0],
            0,
        )

    def test_reset_metrics(self):
        self.profiler.timings["test"] = [1, 2, 3]
        self.profiler.reset_metrics()
        self.assertEqual(self.profiler.timings, {})

    def test_get_duration_timings(self):
        self.profiler.timings = {
            "layer1_forward_start": [0.1, 0.2],
            "layer1_forward_end": [0.3, 0.4],
        }
        duration_timings = self.profiler.get_duration_timings()
        self.assertIn("layer1_forward", duration_timings)
        self.assertEqual(len(duration_timings["layer1_forward"]), 2)

    def test_get_average_metrics(self):
        self.profiler.timings = {
            "layer1_forward_start": [0.1, 0.2, 0.3],
            "layer1_forward_end": [0.4, 0.5, 0.6],
        }
        avg_timings = self.profiler.get_average_metrics(1, 2, ["layer1"])
        self.assertIn("layer1_forward", avg_timings)
        self.assertGreater(avg_timings["layer1_forward"], 0)


class TestLayerMemoryProfiler(unittest.TestCase):
    def setUp(self):
        self.profiler = MemoryProfiler(layer_names=["layer0", "layer1", "layer2"])

    def test_reset_metrics(self):
        self.profiler.activation_memory_usage["layer1"] = [10, 20]
        self.profiler.reset_metrics()
        self.assertEqual(self.profiler.activation_memory_usage["layer1"], [])

    def test_log_activation_memory(self):
        self.profiler.log_activation_memory_info([20.2, 50.5, 30.2])
        self.assertEqual(self.profiler.activation_memory_usage["layer1"], [50.5])
        self.assertEqual(self.profiler.activation_memory_usage["layer2"], [30.2])

    def test_log_weight_grad_optimizer_memory_info(self):

        model = torch.nn.Sequential(
            OrderedDict(
                [
                    ("layer0", torch.nn.Linear(10, 20)),
                    ("layer1", torch.nn.ReLU()),
                    ("layer2", torch.nn.Linear(20, 1000)),
                ]
            )
        )
        model = model.to(torch.device("cuda"))

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # run a forward pass and backward pass for a dummy input to populate the gradients and optimizer states
        dummy_input = torch.randn(4, 10).to(torch.device("cuda"))
        y = torch.tensor([1, 2, 3, 4]).to(torch.device("cuda"))
        pred = model(dummy_input)
        loss = torch.nn.CrossEntropyLoss()(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        self.profiler.log_weight_grad_optimizer_memory_info(
            model, optimizer, torch.device("cuda:0")
        )
        
        
        self.assertGreater(sum(self.profiler.weight_memory_usage["layer2"]), 0)
        self.assertGreater(sum(self.profiler.grad_memory_usage["layer2"]), 0)
        self.assertGreater(sum(self.profiler.optimizer_memory_usage["layer2"]), 0)


class TestModelLayerProfile(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(10, 20), torch.nn.ReLU(), torch.nn.Linear(20, 5)
        )
        self.profiler = ModelProfiler(self.model, layer_names=["0", "2"])

    def test_get_parameters_per_layer(self):
        params = self.profiler.get_parameters_per_layer()
        self.assertEqual(len(params), 2)
        self.assertTrue(all(isinstance(p[1], int) for p in params))

    def test_get_total_parameters(self):
        total_params = self.profiler.get_total_parameters()
        self.assertTrue(total_params > 0)

    def test_get_activation_parameters_per_layer(self):
        dummy_input = torch.randn(1, 10)
        activations = self.profiler.get_activation_parameters_per_layer(dummy_input)
        self.assertEqual(len(activations), 2)
        self.assertTrue(all(isinstance(a[1], int) for a in activations))


class TestUtilityFunctions(unittest.TestCase):
    def test_get_layer_names(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20), torch.nn.ReLU(), torch.nn.Linear(20, 5)
        )
        layer_names = get_layer_names(model)
        self.assertIn("0", layer_names)
        self.assertIn("2", layer_names)

    def test_register_timing_hooks(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20), torch.nn.ReLU(), torch.nn.Linear(20, 5)
        )
        timings = {}
        memory_usage = {}
        hooks = {}
        register_timing_hooks(model, timings, memory_usage, ["0", "2"], hooks=hooks)

        # assert hooks are registered
        self.assertIn("0", hooks)
        self.assertIn("2", hooks)

        # execute forward pass and check if timings are recorded
        dummy_input = torch.randn(1, 10)
        model(dummy_input)
        self.assertIn("0_forward_start", timings)
        self.assertIn("0_forward_end", timings)
        self.assertIn("2_forward_start", timings)
        self.assertIn("2_forward_end", timings)

        # execute backward pass and check if timings are recorded
        loss = torch.nn.CrossEntropyLoss()(model(dummy_input), torch.tensor([1]))
        loss.backward()
        self.assertIn("0_backward_start", timings)
        self.assertIn("0_backward_end", timings)
        self.assertIn("2_backward_start", timings)
        self.assertIn("2_backward_end", timings)

        # check if memory usage is recorded
        self.assertIn("0_forward_start_reserved", memory_usage)
        self.assertIn("2_forward_start_reserved", memory_usage)
        self.assertIn("0_backward_start_reserved", memory_usage)
        self.assertIn("2_backward_start_reserved", memory_usage)
        self.assertIn("0_forward_start_allocated", memory_usage)
        self.assertIn("2_forward_start_allocated", memory_usage)
        self.assertIn("0_backward_start_allocated", memory_usage)
        self.assertIn("2_backward_start_allocated", memory_usage)

    def test_get_param_act_info(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20), torch.nn.ReLU(), torch.nn.Linear(20, 5)
        )
        dummy_input = torch.randn(1, 10)
        layer_names = ["0", "2"]
        info = get_param_act_info("test_model", model, layer_names, dummy_input)
        self.assertEqual(info["model_name"], "test_model")
        self.assertEqual(info["number_of_layers"], len(layer_names))

    def test_measure_activation_shape(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20), torch.nn.ReLU(), torch.nn.Linear(20, 5)
        )
        dummy_input = torch.randn(1, 10)
        activations, activations_size, input_shapes = measure_activation_shape(
            model, ["0", "2"], dummy_input
        )
        self.assertEqual(len(activations), 2)
        self.assertEqual(len(activations_size), 2)
        self.assertEqual(len(input_shapes), 2)


if __name__ == "__main__":
    unittest.main()
