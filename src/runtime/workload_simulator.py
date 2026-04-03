"""Workload simulator for realistic traffic scenarios."""
import torch
import random
import time
import numpy as np


class WorkloadSimulator:
    """Generates requests with arrivals and variable lengths."""
    
    def __init__(self, avg_arrival_rate=1.0, prompt_range=(10, 50), decode_range=(20, 100)):
        self.avg_arrival_rate = avg_arrival_rate # req/sec
        self.prompt_range = prompt_range
        self.decode_range = decode_range
    
    def generate_requests(self, num_requests):
        """Generate a list of request specs with arrival times."""
        requests = []
        curr_time = 0.0
        
        for i in range(num_requests):
            # Poisson arrival (exponential inter-arrival time)
            inter_arrival = random.expovariate(self.avg_arrival_rate)
            curr_time += inter_arrival
            
            prompt_len = random.randint(*self.prompt_range)
            decode_len = random.randint(*self.decode_range)
            
            requests.append({
                "id": i,
                "arrival_time": curr_time,
                "prompt_len": prompt_len,
                "decode_len": decode_len,
            })
        
        return requests


def run_workload(controller, requests, tokenizer=None):
    """Run a simulated workload through the controller."""
    results = []
    
    # Simple sequential execution for now (not overlapping arrivals)
    # This simulates a single-request-at-a-time server
    for req in requests:
        # Simulate waiting for arrival
        # (In a real simulator, we'd use an event-driven loop or async)
        
        # Create dummy prompt tokens
        prompt_ids = torch.randint(1, 100, (1, req["prompt_len"]), device=controller.device)
        
        controller.kv_cache.reset()
        res = controller.generate(prompt_ids, req["decode_len"])
        
        res["request_id"] = req["id"]
        res["arrival_time"] = req["arrival_time"]
        results.append(res)
    
    return results
