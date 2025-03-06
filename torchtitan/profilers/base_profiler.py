from typing import List

class BaseProfiler:
    def __init__(self, layer_names: List[str] = None):
        """Base profiler class for time and memory profiling

        Args:
            layer_names (List[str], optional): List of layer names to profile. Defaults to None.
        """
        self.layer_names = layer_names

    def get_average_metrics(self, warm: int, active: int, layers_name: List[str]):
        """Base method for getting average metrics over warm-up and active steps

        Args:
            warm (int): Number of warm-up steps to skip
            active (int): Number of active steps to average over
            layers_name (List[str]): List of layer names to get metrics for

        Returns:
            dict: Dictionary containing averaged metrics
        """
        assert active > 0, "Active steps should be greater than 0"

    def reset_metrics(self):
        """Reset all metrics to initial state"""
        pass

    def _validate_layer_names(self, recorded_layer_names: List[str], layers_name: List[str]):
        """Validate that all requested layers exist in recorded layers

        Args:
            recorded_layer_names (List[str]): List of layer names that were recorded
            layers_name (List[str]): List of layer names being requested

        Raises:
            AssertionError: If a requested layer is not found in recorded layers
        """
        for ln in layers_name:
            assert ln in recorded_layer_names, f"Layer {ln} not found in the model layers {recorded_layer_names}"
