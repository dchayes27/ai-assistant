#!/bin/bash

# AI Assistant Test Runner
# Comprehensive test execution script with multiple options

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Default values
COVERAGE=false
FAST_ONLY=false
PARALLEL=false
VERBOSE=false
CATEGORY=""
OUTPUT_DIR="test-reports"
BENCHMARK=false
PROFILE=false

# Help function
show_help() {
    cat << EOF
AI Assistant Test Runner

Usage: $0 [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    -c, --coverage          Run with coverage reporting
    -f, --fast              Run only fast tests (exclude slow/performance tests)
    -p, --parallel          Run tests in parallel
    -v, --verbose           Verbose output
    -b, --benchmark         Run performance benchmarks
    --profile              Run with memory profiling
    --unit                 Run only unit tests
    --integration          Run only integration tests
    --performance          Run only performance tests
    --e2e                  Run only end-to-end tests
    --audio                Run only audio tests
    --requires-ollama      Run tests requiring Ollama
    --clean                Clean test artifacts before running
    --output-dir DIR       Output directory for reports (default: test-reports)

EXAMPLES:
    $0                              # Run all tests
    $0 --fast --coverage            # Fast tests with coverage
    $0 --unit --verbose             # Unit tests with verbose output
    $0 --performance --benchmark    # Performance tests with benchmarks
    $0 --parallel --coverage        # All tests in parallel with coverage
    $0 --integration --clean        # Clean integration tests

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--coverage)
            COVERAGE=true
            shift
            ;;
        -f|--fast)
            FAST_ONLY=true
            shift
            ;;
        -p|--parallel)
            PARALLEL=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -b|--benchmark)
            BENCHMARK=true
            shift
            ;;
        --profile)
            PROFILE=true
            shift
            ;;
        --unit)
            CATEGORY="unit"
            shift
            ;;
        --integration)
            CATEGORY="integration"
            shift
            ;;
        --performance)
            CATEGORY="performance"
            shift
            ;;
        --e2e)
            CATEGORY="e2e"
            shift
            ;;
        --audio)
            CATEGORY="audio"
            shift
            ;;
        --requires-ollama)
            CATEGORY="requires_ollama"
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Clean artifacts if requested
if [[ "$CLEAN" == true ]]; then
    print_status "Cleaning test artifacts..."
    rm -rf "$OUTPUT_DIR"/*
    rm -rf .pytest_cache
    rm -rf htmlcov
    rm -rf __pycache__
    find . -name "*.pyc" -delete
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    print_success "Cleaned test artifacts"
fi

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    print_warning "No virtual environment detected. Consider activating venv first:"
    print_warning "  source venv/bin/activate"
fi

# Check dependencies
print_status "Checking dependencies..."
if ! python -c "import pytest" 2>/dev/null; then
    print_error "pytest not found. Please install test dependencies:"
    print_error "  pip install -r requirements.txt"
    exit 1
fi

# Build pytest command
PYTEST_CMD="pytest"

# Add verbosity
if [[ "$VERBOSE" == true ]]; then
    PYTEST_CMD="$PYTEST_CMD -v"
fi

# Add parallel execution
if [[ "$PARALLEL" == true ]]; then
    if python -c "import xdist" 2>/dev/null; then
        PYTEST_CMD="$PYTEST_CMD -n auto"
    else
        print_warning "pytest-xdist not available, running sequentially"
    fi
fi

# Add coverage
if [[ "$COVERAGE" == true ]]; then
    PYTEST_CMD="$PYTEST_CMD --cov=memory --cov=core --cov=mcp_server --cov=gui"
    PYTEST_CMD="$PYTEST_CMD --cov-report=html:$OUTPUT_DIR/htmlcov"
    PYTEST_CMD="$PYTEST_CMD --cov-report=xml:$OUTPUT_DIR/coverage.xml"
    PYTEST_CMD="$PYTEST_CMD --cov-report=term-missing"
fi

# Add category filtering
if [[ -n "$CATEGORY" ]]; then
    PYTEST_CMD="$PYTEST_CMD -m $CATEGORY"
fi

# Add fast-only filtering
if [[ "$FAST_ONLY" == true ]]; then
    if [[ -n "$CATEGORY" ]]; then
        PYTEST_CMD="$PYTEST_CMD and not slow and not performance"
    else
        PYTEST_CMD="$PYTEST_CMD -m 'not slow and not performance'"
    fi
fi

# Add benchmark support
if [[ "$BENCHMARK" == true ]]; then
    if python -c "import pytest_benchmark" 2>/dev/null; then
        PYTEST_CMD="$PYTEST_CMD --benchmark-json=$OUTPUT_DIR/benchmark.json"
    else
        print_warning "pytest-benchmark not available, skipping benchmark output"
    fi
fi

# Add memory profiling
if [[ "$PROFILE" == true ]]; then
    if python -c "import memory_profiler" 2>/dev/null; then
        PYTEST_CMD="$PYTEST_CMD --profile"
    else
        print_warning "memory_profiler not available, skipping profiling"
    fi
fi

# Add output files
PYTEST_CMD="$PYTEST_CMD --junitxml=$OUTPUT_DIR/test-results.xml"

# Show test environment info
print_status "Test Environment Information:"
echo "  Python version: $(python --version)"
echo "  Pytest version: $(pytest --version | head -n1)"
echo "  Working directory: $(pwd)"
echo "  Output directory: $OUTPUT_DIR"
echo "  Virtual environment: ${VIRTUAL_ENV:-None}"

# Show what tests will run
print_status "Test Configuration:"
echo "  Coverage: $COVERAGE"
echo "  Fast only: $FAST_ONLY"
echo "  Parallel: $PARALLEL"
echo "  Verbose: $VERBOSE"
echo "  Category: ${CATEGORY:-all}"
echo "  Benchmark: $BENCHMARK"
echo "  Profile: $PROFILE"

# Check external services
print_status "Checking external services..."

# Check Ollama
if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    print_success "Ollama service is available"
    OLLAMA_AVAILABLE=true
else
    print_warning "Ollama service not available - related tests will be skipped"
    OLLAMA_AVAILABLE=false
fi

# Check audio capabilities
if python -c "import sounddevice, whisper" 2>/dev/null; then
    print_success "Audio libraries available"
    AUDIO_AVAILABLE=true
else
    print_warning "Audio libraries not available - audio tests will be skipped"
    AUDIO_AVAILABLE=false
fi

# Modify pytest command based on available services
if [[ "$OLLAMA_AVAILABLE" == false ]] && [[ "$CATEGORY" != "requires_ollama" ]]; then
    if [[ "$PYTEST_CMD" == *"-m"* ]]; then
        PYTEST_CMD="$PYTEST_CMD and not requires_ollama"
    else
        PYTEST_CMD="$PYTEST_CMD -m 'not requires_ollama'"
    fi
fi

if [[ "$AUDIO_AVAILABLE" == false ]] && [[ "$CATEGORY" != "audio" ]] && [[ "$CATEGORY" != "requires_audio" ]]; then
    if [[ "$PYTEST_CMD" == *"-m"* ]]; then
        PYTEST_CMD="$PYTEST_CMD and not requires_audio"
    else
        PYTEST_CMD="$PYTEST_CMD -m 'not requires_audio'"
    fi
fi

# Run the tests
print_status "Running tests..."
echo "Command: $PYTEST_CMD"
echo ""

# Record start time
START_TIME=$(date +%s)

# Execute tests
if eval "$PYTEST_CMD"; then
    TEST_RESULT="PASSED"
    EXIT_CODE=0
else
    TEST_RESULT="FAILED"
    EXIT_CODE=1
fi

# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Show results
echo ""
print_status "Test Results Summary:"
echo "  Status: $TEST_RESULT"
echo "  Duration: ${DURATION}s"
echo "  Output directory: $OUTPUT_DIR"

if [[ "$COVERAGE" == true ]]; then
    echo "  Coverage report: $OUTPUT_DIR/htmlcov/index.html"
fi

if [[ "$BENCHMARK" == true ]]; then
    echo "  Benchmark results: $OUTPUT_DIR/benchmark.json"
fi

# List generated files
if [[ -d "$OUTPUT_DIR" ]]; then
    echo ""
    print_status "Generated files:"
    ls -la "$OUTPUT_DIR/"
fi

# Final status
if [[ $EXIT_CODE -eq 0 ]]; then
    print_success "All tests completed successfully!"
else
    print_error "Some tests failed. Check the output above for details."
fi

# Show coverage summary if available
if [[ "$COVERAGE" == true ]] && [[ -f "$OUTPUT_DIR/coverage.xml" ]]; then
    echo ""
    print_status "Coverage Summary:"
    if command -v coverage >/dev/null 2>&1; then
        coverage report --show-missing 2>/dev/null | tail -n 5 || true
    fi
fi

# Performance summary
if [[ "$BENCHMARK" == true ]] && [[ -f "$OUTPUT_DIR/benchmark.json" ]]; then
    echo ""
    print_status "Performance Summary:"
    python -c "
import json
try:
    with open('$OUTPUT_DIR/benchmark.json') as f:
        data = json.load(f)
    if 'benchmarks' in data:
        print(f'  Benchmarks run: {len(data[\"benchmarks\"])}')
        if data['benchmarks']:
            times = [b['stats']['mean'] for b in data['benchmarks']]
            print(f'  Average time: {sum(times)/len(times):.3f}s')
            print(f'  Fastest test: {min(times):.3f}s')
            print(f'  Slowest test: {max(times):.3f}s')
except Exception:
    pass
" 2>/dev/null || true
fi

# Exit with appropriate code
exit $EXIT_CODE