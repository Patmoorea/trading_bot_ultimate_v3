class LoadTester:
    def __init__(self):
        self.scenarios = {
            'high_frequency': self.test_hf_trading,
            'multi_pair': self.test_multi_pair,
            'news_impact': self.test_news_processing
        }
        
    async def run_all_tests(self):
        results = {}
        for name, test in self.scenarios.items():
            results[name] = await test()
        return results
