import time
from datetime import datetime


class PerformanceTracker:
    def __init__(self):
        self.metrics = {"execution_times": [], "opportunities_found": 0}

    def log_execution(self, duration: float):
        self.metrics["execution_times"].append(duration)
        if len(self.metrics["execution_times"]) > 100:
            self.metrics["execution_times"].pop(0)

    def log_opportunity(self):
        self.metrics["opportunities_found"] += 1

    def get_stats(self):
        avg_time = (
            sum(self.metrics["execution_times"]) /
            len(self.metrics["execution_times"])
            if self.metrics["execution_times"]
            else 0
        )
        return {
            "avg_execution_time": avg_time,
            "total_opportunities": self.metrics["opportunities_found"],
            "timestamp": datetime.now().isoformat(),
        }
