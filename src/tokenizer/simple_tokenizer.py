"""Simple word-level tokenizer."""
from collections import Counter


class SimpleTokenizer:
    """Basic word tokenizer for demo purposes."""
    
    def __init__(self, vocab=None):
        self.vocab = vocab or {"<PAD>": 0, "<UNK>": 1}
        self.inv_vocab = {i: t for t, i in self.vocab.items()}
    
    def build_vocab(self, corpus, max_vocab=100):
        """Build vocabulary from text corpus."""
        words = corpus.split()
        counts = Counter(words)
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        for word, _ in counts.most_common(max_vocab):
            self.vocab[word] = len(self.vocab)
        self.inv_vocab = {i: t for t, i in self.vocab.items()}
        return self
    
    def encode(self, text):
        """Convert text to token IDs."""
        return [self.vocab.get(tok, self.vocab["<UNK>"]) for tok in text.split()]
    
    def decode(self, token_ids):
        """Convert token IDs back to text."""
        return " ".join([self.inv_vocab.get(i, "<UNK>") for i in token_ids])
    
    def __len__(self):
        return len(self.vocab)
