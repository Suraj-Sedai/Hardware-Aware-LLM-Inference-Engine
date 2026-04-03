"""Continuous batching and request management."""


class ContinuousBatching:
    """Manages active requests and dynamically adds new ones."""
    
    def __init__(self, max_batch_size):
        self.max_batch_size = max_batch_size
        self.active_requests = []
    
    def can_add(self):
        return len(self.active_requests) < self.max_batch_size
    
    def add_request(self, request):
        if self.can_add():
            self.active_requests.append(request)
            return True
        return False
    
    def remove_finished(self):
        self.active_requests = [r for r in self.active_requests if not r.is_finished()]
