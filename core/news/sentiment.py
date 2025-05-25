from transformers import BertTokenizer, BertForSequenceClassification
import torch

class NewsAnalyzer:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        self.model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
        
    def analyze(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        return torch.softmax(outputs.logits, dim=1).detach().numpy()

class EnhancedNewsAnalyzer(NewsAnalyzer):
    """Nouvelle version avec gestion des erreurs améliorée"""
    
    def __init__(self):
        super().__init__()
        self.cache = {}
        self.timeout = 5.0
    
    def analyze_batch(self, texts):
        """Analyse par lot pour meilleure performance"""
        try:
            inputs = self.tokenizer(
                texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
            return torch.softmax(outputs.logits, dim=1).numpy()
        except Exception as e:
            logger.error(f"Analyse batch échouée: {str(e)}")
            return None
