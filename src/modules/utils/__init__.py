"""
Utils module initialization
Version 1.0.0 - Created: 2025-05-26 14:31:40 by Patmoorea
"""

# Common utility functions
def format_timestamp(timestamp):
    """Format timestamp to standard format"""
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")

def parse_timestamp(timestamp_str):
    """Parse timestamp from standard format"""
    from datetime import datetime
    return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
