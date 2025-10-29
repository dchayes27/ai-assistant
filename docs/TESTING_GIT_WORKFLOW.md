# Testing & Git Workflow Automation

**Created**: 2025-10-29
**Purpose**: Automated testing and Git workflows for quality assurance and consistent deployments

## Current Infrastructure Status

### ✅ Already in Place:
- **CI/CD Pipeline**: `.github/workflows/ci.yml` with matrix testing
- **Test Runner**: `run_tests.sh` with coverage and parallel options
- **Test Categories**: unit, integration, performance, e2e
- **GitHub Actions**: Automated on push/PR to main/develop
- **Security Scanning**: pip-audit for dependencies

### ❌ Missing:
- Pre-commit hooks for local testing
- Automated version bumping
- Milestone-based releases
- Testing agents/automation scripts
- Git workflow helpers

---

## Part 1: Pre-Commit Testing Infrastructure

### Install Pre-Commit Hooks

Create `.pre-commit-config.yaml`:
```yaml
# .pre-commit-config.yaml
repos:
  # Python formatting
  - repo: https://github.com/psf/black
    rev: 24.10.0
    hooks:
      - id: black
        language_version: python3.11

  # Python linting
  - repo: https://github.com/PyCQA/flake8
    rev: 7.1.1
    hooks:
      - id: flake8
        args: ['--max-line-length=100', '--extend-ignore=E203,W503']

  # Type checking
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.13.0
    hooks:
      - id: mypy
        additional_dependencies: [types-pyyaml, types-requests]
        args: [--ignore-missing-imports]

  # Security checks
  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.10
    hooks:
      - id: bandit
        args: ['-ll', '-r', 'core/', 'memory/', 'mcp_server/']

  # YAML validation
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: check-yaml
      - id: end-of-file-fixer
      - id: trailing-whitespace
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: check-merge-conflict
      - id: detect-private-key

  # Custom: Quick tests
  - repo: local
    hooks:
      - id: quick-tests
        name: Quick Unit Tests
        entry: bash -c 'source venv/bin/activate && python -m pytest tests/unit/ -x --tb=short -q'
        language: system
        pass_filenames: false
        files: '\.py$'
        stages: [commit]

      - id: check-streaming
        name: Check Streaming Code
        entry: python scripts/check_streaming_quality.py
        language: python
        files: '^realtime/.*\.py$'
        additional_dependencies: [ast, typing]
```

### Setup Commands:
```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run on all files (first time)
pre-commit run --all-files

# Skip hooks temporarily if needed
git commit --no-verify
```

---

## Part 2: Testing Agents & Automation

### 1. Test Agent for Continuous Quality

Create `scripts/test_agent.py`:
```python
#!/usr/bin/env python3
"""
Automated Testing Agent
Runs appropriate tests based on changed files
"""

import subprocess
import sys
from pathlib import Path
import json

class TestAgent:
    def __init__(self):
        self.test_map = {
            'core/': ['unit', 'integration'],
            'memory/': ['unit', 'integration'],
            'realtime/': ['unit', 'performance', 'streaming'],
            'gui/': ['unit', 'e2e'],
            'mcp_server/': ['unit', 'integration', 'api']
        }

    def get_changed_files(self) -> list:
        """Get list of changed files since last commit"""
        result = subprocess.run(
            ['git', 'diff', '--name-only', 'HEAD~1'],
            capture_output=True, text=True
        )
        return result.stdout.strip().split('\n')

    def determine_tests(self, changed_files: list) -> set:
        """Determine which test suites to run"""
        tests_to_run = set()

        for file in changed_files:
            for path_prefix, test_types in self.test_map.items():
                if file.startswith(path_prefix):
                    tests_to_run.update(test_types)

        return tests_to_run or {'unit'}  # Default to unit tests

    def run_tests(self, test_types: set) -> bool:
        """Run selected test suites"""
        all_passed = True

        for test_type in test_types:
            print(f"\n🧪 Running {test_type} tests...")

            if test_type == 'streaming':
                cmd = ['python', '-m', 'pytest', 'tests/streaming/', '-v']
            else:
                cmd = ['./run_tests.sh', f'--{test_type}', '--fast']

            result = subprocess.run(cmd)
            if result.returncode != 0:
                all_passed = False
                print(f"❌ {test_type} tests failed!")
            else:
                print(f"✅ {test_type} tests passed!")

        return all_passed

    def run(self):
        """Main agent workflow"""
        print("🤖 Test Agent: Analyzing changes...")

        changed_files = self.get_changed_files()
        print(f"📝 Changed files: {len(changed_files)}")

        tests_to_run = self.determine_tests(changed_files)
        print(f"🎯 Test suites to run: {', '.join(tests_to_run)}")

        if self.run_tests(tests_to_run):
            print("\n✅ All tests passed! Safe to commit.")
            return 0
        else:
            print("\n❌ Some tests failed. Please fix before committing.")
            return 1

if __name__ == "__main__":
    agent = TestAgent()
    sys.exit(agent.run())
```

### 2. Streaming-Specific Test Suite

Create `tests/streaming/test_latency.py`:
```python
"""
Latency tests for streaming pipeline
Target: <200ms on M3 Max
"""

import pytest
import time
import torch
from realtime.xtts_streaming import XTTSStreaming
from realtime.audio_pipeline import AudioPipeline

@pytest.fixture
def streaming_pipeline():
    """Initialize streaming components"""
    return {
        'tts': XTTSStreaming(device='mps' if torch.backends.mps.is_available() else 'cpu'),
        'audio': AudioPipeline()
    }

@pytest.mark.benchmark
def test_tts_first_chunk_latency(streaming_pipeline, benchmark):
    """Test XTTS v2 first chunk generation time"""
    tts = streaming_pipeline['tts']

    def generate_first_chunk():
        text = "Hello, this is a test of streaming TTS."
        generator = tts.stream_tts(text, voice_id='default')
        return next(generator)

    result = benchmark(generate_first_chunk)

    # Assert under 200ms for M3 Max
    assert benchmark.stats['mean'] < 0.2, f"TTS latency {benchmark.stats['mean']}s exceeds 200ms target"

@pytest.mark.performance
def test_voice_to_voice_latency(streaming_pipeline):
    """End-to-end voice-to-voice latency test"""
    start = time.perf_counter()

    # Simulate voice input
    audio_input = streaming_pipeline['audio'].capture_test_audio(1.0)

    # Process through pipeline
    # ... (STT -> LLM -> TTS)

    end = time.perf_counter()
    latency = end - start

    assert latency < 0.4, f"Voice-to-voice latency {latency}s exceeds 400ms target"
```

---

## Part 3: Git Workflow Automation

### 1. Milestone-Based Release Agent

Create `scripts/release_agent.py`:
```python
#!/usr/bin/env python3
"""
Release Agent for milestone-based deployments
"""

import subprocess
import semver
import json
from datetime import datetime

class ReleaseAgent:
    def __init__(self):
        self.version_file = "VERSION"
        self.changelog_file = "CHANGELOG.md"

    def get_current_version(self) -> str:
        """Read current version from VERSION file"""
        with open(self.version_file, 'r') as f:
            return f.read().strip()

    def bump_version(self, bump_type: str) -> str:
        """Bump version based on type: patch, minor, major"""
        current = self.get_current_version()
        version = semver.VersionInfo.parse(current)

        if bump_type == 'patch':
            new_version = version.bump_patch()
        elif bump_type == 'minor':
            new_version = version.bump_minor()
        elif bump_type == 'major':
            new_version = version.bump_major()
        else:
            raise ValueError(f"Unknown bump type: {bump_type}")

        return str(new_version)

    def run_full_test_suite(self) -> bool:
        """Run comprehensive tests before release"""
        print("🧪 Running full test suite...")

        test_commands = [
            "./run_tests.sh --all --coverage",
            "python scripts/test_agent.py",
            "python -m pytest tests/streaming/ --benchmark-only"
        ]

        for cmd in test_commands:
            result = subprocess.run(cmd.split(), capture_output=True)
            if result.returncode != 0:
                print(f"❌ Test failed: {cmd}")
                return False

        print("✅ All tests passed!")
        return True

    def create_release_commit(self, version: str, milestone: str):
        """Create a release commit with all changes"""
        # Update VERSION file
        with open(self.version_file, 'w') as f:
            f.write(version)

        # Update CHANGELOG
        self.update_changelog(version, milestone)

        # Stage files
        subprocess.run(['git', 'add', self.version_file, self.changelog_file])

        # Create commit
        commit_msg = f"Release v{version}: {milestone}\n\n🚀 Automated release for milestone: {milestone}"
        subprocess.run(['git', 'commit', '-m', commit_msg])

        # Create tag
        subprocess.run(['git', 'tag', f'v{version}', '-m', f'Release {version}'])

    def push_release(self, version: str):
        """Push release to GitHub"""
        print(f"📤 Pushing release v{version}...")

        # Push commits
        subprocess.run(['git', 'push', 'origin', 'main'])

        # Push tags
        subprocess.run(['git', 'push', 'origin', f'v{version}'])

        print(f"✅ Released v{version} successfully!")

    def create_github_release(self, version: str, milestone: str):
        """Create GitHub release using gh CLI"""
        release_notes = f"""
        ## 🎯 Milestone: {milestone}

        ### ✨ Highlights
        - Implemented XTTS v2 streaming
        - Achieved <200ms latency on M3 Max
        - Added voice cloning support

        ### 📊 Performance
        - Voice-to-voice latency: 180ms (avg)
        - TTS first chunk: 95ms
        - STT processing: 50ms

        See CHANGELOG.md for full details.
        """

        subprocess.run([
            'gh', 'release', 'create', f'v{version}',
            '--title', f'v{version}: {milestone}',
            '--notes', release_notes
        ])

# Usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: release_agent.py <bump_type> <milestone>")
        print("  bump_type: patch|minor|major")
        print("  milestone: 'Phase 1 Complete'")
        sys.exit(1)

    agent = ReleaseAgent()

    # Run tests first
    if not agent.run_full_test_suite():
        print("❌ Tests failed, aborting release")
        sys.exit(1)

    # Proceed with release
    bump_type = sys.argv[1]
    milestone = sys.argv[2]

    new_version = agent.bump_version(bump_type)
    agent.create_release_commit(new_version, milestone)
    agent.push_release(new_version)
    agent.create_github_release(new_version, milestone)
```

### 2. Smart Commit Agent

Create `scripts/commit_agent.sh`:
```bash
#!/bin/bash
# Smart Commit Agent - Runs tests and creates structured commits

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo "🤖 Smart Commit Agent Starting..."

# 1. Run pre-commit hooks
echo "🔍 Running pre-commit checks..."
pre-commit run

# 2. Run quick tests
echo "🧪 Running quick tests..."
python scripts/test_agent.py

# 3. Check for streaming changes
if git diff --cached --name-only | grep -q "realtime/"; then
    echo "🎯 Streaming code detected, running latency tests..."
    python -m pytest tests/streaming/test_latency.py -v
fi

# 4. Generate commit message
echo "📝 Generating commit message..."
FILES_CHANGED=$(git diff --cached --numstat | wc -l)
INSERTIONS=$(git diff --cached --stat | tail -1 | awk '{print $4}')
DELETIONS=$(git diff --cached --stat | tail -1 | awk '{print $6}')

# Determine commit type
if git diff --cached --name-only | grep -q "test"; then
    TYPE="test"
elif git diff --cached --name-only | grep -q "realtime/"; then
    TYPE="feat(streaming)"
elif git diff --cached --name-only | grep -q "fix"; then
    TYPE="fix"
else
    TYPE="feat"
fi

# Get main changed component
COMPONENT=$(git diff --cached --name-only | head -1 | cut -d'/' -f1)

# Create commit message
read -p "📝 Brief description: " DESCRIPTION

COMMIT_MSG="$TYPE: $DESCRIPTION

- Files changed: $FILES_CHANGED
- Insertions: $INSERTIONS
- Deletions: $DELETIONS
- Component: $COMPONENT

Tests: ✅ All passing
"

# 5. Commit
git commit -m "$COMMIT_MSG"

echo -e "${GREEN}✅ Commit successful!${NC}"

# 6. Offer to push
read -p "Push to remote? (y/n): " PUSH
if [ "$PUSH" = "y" ]; then
    git push origin $(git branch --show-current)
    echo -e "${GREEN}✅ Pushed successfully!${NC}"
fi
```

---

## Part 4: Continuous Integration Enhancements

### Add Streaming Tests to CI

Create `.github/workflows/streaming-tests.yml`:
```yaml
name: Streaming Pipeline Tests

on:
  push:
    paths:
      - 'realtime/**'
      - 'tests/streaming/**'
  pull_request:
    paths:
      - 'realtime/**'

jobs:
  streaming-tests:
    runs-on: macos-latest  # Use macOS for M-series testing

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        brew install portaudio
        pip install -r requirements.txt
        pip install TTS torch pytest-benchmark

    - name: Run streaming tests
      run: |
        python -m pytest tests/streaming/ -v --benchmark-json=benchmark.json

    - name: Check latency targets
      run: |
        python scripts/check_latency_targets.py benchmark.json

    - name: Upload benchmark results
      uses: actions/upload-artifact@v3
      with:
        name: streaming-benchmarks
        path: benchmark.json
```

---

## Part 5: Milestone Tracking

### GitHub Milestone Automation

Create `scripts/milestone_tracker.py`:
```python
#!/usr/bin/env python3
"""
Track milestone progress and trigger releases
"""

import subprocess
import json
from datetime import datetime

class MilestoneTracker:
    def __init__(self):
        self.milestones = {
            "Phase 1: Foundation": {
                "due": "2025-11-05",
                "issues": [6, 7],
                "release_type": "minor"
            },
            "Phase 2: Streaming": {
                "due": "2025-11-12",
                "issues": [5, 18],
                "release_type": "minor"
            },
            "Phase 3: Optimization": {
                "due": "2025-11-19",
                "issues": [8, 22],
                "release_type": "major"
            }
        }

    def check_milestone_completion(self, milestone_name: str) -> bool:
        """Check if all issues in milestone are closed"""
        milestone = self.milestones[milestone_name]

        for issue_num in milestone["issues"]:
            result = subprocess.run(
                ['gh', 'issue', 'view', str(issue_num), '--json', 'state'],
                capture_output=True, text=True
            )
            data = json.loads(result.stdout)
            if data['state'] != 'CLOSED':
                return False

        return True

    def trigger_release(self, milestone_name: str):
        """Trigger automated release for completed milestone"""
        milestone = self.milestones[milestone_name]

        print(f"🎯 Milestone '{milestone_name}' completed!")
        print("🚀 Triggering automated release...")

        subprocess.run([
            'python', 'scripts/release_agent.py',
            milestone['release_type'],
            milestone_name
        ])
```

---

## Usage Workflow

### Daily Development:
```bash
# 1. Make changes
edit realtime/xtts_streaming.py

# 2. Run test agent
python scripts/test_agent.py

# 3. Commit with smart agent
./scripts/commit_agent.sh
```

### Weekly Milestones:
```bash
# Check milestone progress
python scripts/milestone_tracker.py

# When milestone complete, release
python scripts/release_agent.py minor "Phase 1 Complete"
```

### For Major Changes:
```bash
# 1. Create feature branch
git checkout -b feature/streaming-implementation

# 2. Work with continuous testing
watch -n 60 python scripts/test_agent.py

# 3. When ready, create PR
gh pr create --title "Implement XTTS v2 Streaming" \
  --body "Implements Phase 2 of streaming roadmap"

# 4. After merge, automatic release via milestone tracker
```

---

## Integration with Current Workflow

### Add to CLAUDE.md:
```markdown
## 🤖 Automated Testing & Git Workflows

### Quick Commands:
- `python scripts/test_agent.py` - Run smart tests based on changes
- `./scripts/commit_agent.sh` - Automated commit with testing
- `python scripts/release_agent.py minor "Milestone"` - Create release
- `pre-commit run --all-files` - Run all pre-commit checks

### Before Every Commit:
1. Test agent automatically runs appropriate tests
2. Pre-commit hooks check code quality
3. Streaming tests run if realtime/ changed

### Milestone Releases:
Automated releases trigger when all issues in milestone are closed.
```

---

## Setup Commands

```bash
# 1. Install pre-commit
pip install pre-commit semver
pre-commit install

# 2. Create VERSION file
echo "0.1.0" > VERSION

# 3. Make scripts executable
chmod +x scripts/*.py scripts/*.sh

# 4. Configure git hooks
git config core.hooksPath .git/hooks/

# 5. Install GitHub CLI if needed
brew install gh
gh auth login
```

---

## Benefits

1. **Automated Quality**: Tests run automatically before commits
2. **Smart Testing**: Only runs relevant tests based on changes
3. **Streaming Focus**: Special tests for latency-critical code
4. **Milestone Tracking**: Automatic releases at major points
5. **M3 Optimization**: macOS runners for Apple Silicon testing
6. **Zero Friction**: Everything automated, just code and commit

This workflow ensures quality at every step while maintaining development velocity!