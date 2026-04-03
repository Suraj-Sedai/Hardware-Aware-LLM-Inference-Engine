"""Latency tracking for phases and layers."""
import time
import torch


class LatencyTracker:
    """Tracks latency for different phases and layers using CUDA events."""
    
    def __init__(self, device):
        self.device = device
        self.use_cuda = device == "cuda" and torch.cuda.is_available()
        self.phase_times = {}
        self.layer_times = {}
        self.events = {}
    
    def start_phase(self, phase_name):
        if self.use_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            self.events[phase_name] = (start_event, end_event)
        else:
            self.phase_times[phase_name] = time.perf_counter()
    
    def end_phase(self, phase_name):
        if self.use_cuda:
            start_event, end_event = self.events[phase_name]
            end_event.record()
        else:
            start_time = self.phase_times[phase_name]
            self.phase_times[phase_name] = (time.perf_counter() - start_time) * 1000 # ms
    
    def start_layer(self, layer_id):
        name = f"layer_{layer_id}"
        self.start_phase(name)
    
    def end_layer(self, layer_id):
        name = f"layer_{layer_id}"
        self.end_phase(name)
    
    def synchronize(self):
        if self.use_cuda:
            torch.cuda.synchronize()
            for name, (start, end) in self.events.items():
                self.phase_times[name] = start.elapsed_time(end) # ms
    
    def get_times(self):
        return self.phase_times
