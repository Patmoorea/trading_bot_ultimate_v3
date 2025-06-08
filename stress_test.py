from analysis.sentiment import NewsSentimentAnalyzer
import time

analyzer = NewsSentimentAnalyzer()
texts = ["BTC up 5%", "Market crash imminent", "Fed raises rates"] * 100

start = time.time()
results = [analyzer.analyze(text) for text in texts]
duration = time.time() - start

print(f"Processed {len(texts)} texts in {duration:.2f}s")
print(f"Avg {duration/len(texts)*1000:.2f}ms per request")
