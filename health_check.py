import unittest
from src.tests.run_all_tests import run_all_tests

class SystemHealthCheck(unittest.TestCase):
    def test_system_readiness(self):
        test_results = run_all_tests()
        self.assertGreaterEqual(test_results['coverage'], 0.92)
        self.assertEqual(test_results['errors'], 0)
        self.assertLess(test_results['latency'], 20)

if __name__ == '__main__':
    unittest.main()
