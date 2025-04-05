#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Start Flask app in background
echo "Starting Flask app..."
ENVIRONMENT=dev python app.py &
APP_PID=$!

# Wait for app to start
sleep 2

# Function to make a request and check response
test_request() {
    local test_name=$1
    local payload=$2
    local expected_status=$3

    echo -e "\nTesting: ${GREEN}$test_name${NC}"
    response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$payload" \
        http://localhost:5000/generate-plan)
    
    status=$(echo $response | jq -r '.status')
    
    if [ "$status" = "$expected_status" ]; then
        echo -e "${GREEN}✓ Test passed${NC}"
    else
        echo -e "${RED}✗ Test failed${NC}"
        echo "Expected status: $expected_status"
        echo "Got response: $response"
    fi
}

# Test 1: Valid request
echo -e "\n${GREEN}Running tests...${NC}"
test_request "Valid request" '{
    "user_doc_format": "test format",
    "user_question": "test question",
    "user_requirements": ["req1", "req2"],
    "additional_context": "test context"
}' "success"

# Test 2: Missing required field
test_request "Missing required field" '{
    "user_doc_format": "test format"
}' "error"

# Test 3: Invalid JSON
test_request "Invalid JSON" '{invalid json}' "error"


# Kill the Flask app
echo -e "\nStopping Flask app..."
kill $APP_PID

echo -e "\n${GREEN}All tests completed${NC}" 