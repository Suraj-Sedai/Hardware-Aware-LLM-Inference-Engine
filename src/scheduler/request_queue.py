"""Request queue for managing incoming inference requests."""
from collections import deque


class RequestQueue:
    """FIFO queue for managing inference requests."""
    
    def __init__(self):
        self.queue = deque()
    
    def add_request(self, request):
        self.queue.append(request)
    
    def pop_request(self):
        if self.queue:
            return self.queue.popleft()
        return None
    
    def size(self):
        return len(self.queue)
