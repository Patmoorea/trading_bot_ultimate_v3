#!/bin/bash
# Updated: 2025-05-27 16:29:55 UTC
# By: Patmoorea

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Starting test suite...${NC}"
echo "Timestamp: 2025-05-27 16:29:55 UTC"
echo "User: Patmoorea"
echo "Environment: Apple M4, macOS 15.3.2"

# Environment checks
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${RED}Error: Virtual environment not activated${NC}"
    echo "Please run: source .venv/bin/activate"
    exit 1
fi

# Cleanup
echo -e "${YELLOW}Cleaning up...${NC}"
find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null
find . -type f -name "*.pyc" -delete
find . -type d -name ".pytest_cache" -exec rm -r {} + 2>/dev/null

# Run tests
echo -e "${GREEN}Running tests...${NC}"

# Basic tests first
python -m pytest tests_new/test_minimal -v

test_status=$?

if [ $test_status -eq 0 ]; then
    echo -e "${GREEN}Basic tests passed successfully!${NC}"
    echo -e "${YELLOW}Now running full test suite...${NC}"
    
    # Full test suite
    python -m pytest tests_new/ -v
    test_status=$?
fi

if [ $test_status -eq 0 ]; then
    echo -e "${GREEN}All tests passed successfully!${NC}"
else
    echo -e "${RED}Some tests failed!${NC}"
fi

exit $test_status
