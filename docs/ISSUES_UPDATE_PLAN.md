# GitHub Issues Consolidation & Update Plan

**Created**: 2025-10-29
**Purpose**: Consolidate and update GitHub issues based on holistic implementation plan

## Issues to CLOSE (No Longer Relevant or Already Fixed)

### Already Completed (from ROADMAP.md):
- ❌ **Close**: Hardcoded Development API Key Exposure (Fixed in PR #4)
- ❌ **Close**: JWT Secret Key Regeneration on Restart (Fixed in PR #4)
- ❌ **Close**: Missing scikit-learn Dependency (Fixed in PR #4)
- ❌ **Close**: No Input Validation on Embedding Imports (Fixed in PR #4)
- ❌ **Close**: No Rate Limiting on Authentication (Fixed in PR #4)
- ❌ **Close**: Print Statements in Production Code (Fixed in PR #4)

### No Longer Relevant with New Architecture:
- ❌ **Close Issue #15**: "Reduce Code Duplication in Tool Manager" - Will be refactored entirely with streaming
- ❌ **Close Issue #10**: "WebSocket Authentication Enforcement" - Being redesigned for streaming WebSocket

---

## Issues to UPDATE with New Scope

### Issue #18: Real-Time Streaming Voice Pipeline
**Current**: Generic streaming request
**Update to**:
```markdown
## Implement XTTS v2 Streaming Pipeline with M3 Optimization

### Objectives:
- [ ] Integrate XTTS v2 with MPS backend for M3 Max
- [ ] Achieve <200ms voice-to-voice latency
- [ ] Implement voice cloning with 10-second samples
- [ ] Add streaming audio with gapless playback
- [ ] Create fallback chain: XTTS v2 → Edge-TTS → pyttsx3

### Technical Requirements:
- Use Metal Performance Shaders for acceleration
- Leverage 64GB unified memory for zero-copy operations
- Implement VAD with Faster-Whisper
- Stream from Ollama with token-by-token processing

### Files to Create:
- `realtime/xtts_streaming.py`
- `realtime/audio_pipeline.py`
- `realtime/streaming_assistant.py`
- `realtime/performance_monitor.py`

### Hardware Target:
- Platform: Apple Silicon M3 Max
- Expected latency: 200ms
- Memory usage: <32GB
```

### Issue #5: Refactor Large Files
**Current**: Generic refactoring request
**Update to**:
```markdown
## Modularize Core Components for Streaming Architecture

### smart_assistant.py (979 lines → 4 modules):
- [ ] Extract audio processing → `core/audio_processor.py`
- [ ] Extract state management → `core/conversation_state.py`
- [ ] Extract metrics → `core/metrics.py`
- [ ] Keep orchestration in `smart_assistant.py` (~300 lines)

### interface.py (1080 lines → modular structure):
- [ ] Create `gui/streaming_tab.py` for new streaming UI
- [ ] Extract visualizers → `gui/components/audio_visualizer.py`
- [ ] Extract state → `gui/app_state.py`
- [ ] Modularize tabs → `gui/tabs/` directory

### Integration with Streaming:
- Ensure new structure supports both batch and streaming modes
- Add feature flags for gradual migration
```

### Issue #6: Add Database Transaction Management
**Current**: Generic transaction request
**Update to**:
```markdown
## Implement Transaction Management for Streaming Operations

### Requirements:
- [ ] Add transaction context manager with rollback
- [ ] Handle streaming conversation updates atomically
- [ ] Implement optimistic locking for real-time updates
- [ ] Add retry logic for streaming interruptions

### Streaming-Specific Needs:
- Transaction batching for token-by-token updates
- Checkpoint system for long streaming sessions
- Non-blocking writes during audio streaming
```

---

## NEW Issues to Create

### Issue: Apple Silicon Optimization Layer
```markdown
## Title: Add M3 Max Hardware Acceleration Support

### Description:
Implement hardware-specific optimizations for Apple Silicon M3 Max including MPS backend, Neural Engine, and unified memory utilization.

### Tasks:
- [ ] Add MPS device detection and fallback
- [ ] Implement CoreML model conversion for Whisper
- [ ] Optimize memory allocation for unified architecture
- [ ] Add Metal Performance HUD integration
- [ ] Create efficiency/performance core task distribution

### Priority: High
### Labels: enhancement, performance, apple-silicon
```

### Issue: Voice Cloning System
```markdown
## Title: Implement Voice Cloning with XTTS v2

### Description:
Add voice cloning capabilities using XTTS v2, allowing users to create custom voice profiles from 10-second samples.

### Tasks:
- [ ] Create voice profile management system
- [ ] Implement speaker embedding generation
- [ ] Add voice sample validation (quality, length)
- [ ] Create UI for voice recording/upload
- [ ] Add voice profile storage and retrieval

### Priority: Medium
### Labels: feature, audio, xtts
```

### Issue: Streaming Performance Dashboard
```markdown
## Title: Real-Time Performance Monitoring Dashboard

### Description:
Create comprehensive monitoring for streaming pipeline including latency tracking, resource usage, and quality metrics.

### Tasks:
- [ ] Add voice-to-voice latency measurement
- [ ] Implement token generation speed tracking
- [ ] Create audio quality scoring
- [ ] Add GPU/Neural Engine utilization monitoring
- [ ] Build Gradio dashboard component

### Priority: Medium
### Labels: monitoring, ui, performance
```

---

## Issues Priority Matrix

### Critical (Week 1):
1. **Issue #7**: Fix HTTP Client Resource Leaks ⚠️
2. **Issue #6**: Database Transaction Management (Updated) ⚠️
3. **NEW**: Apple Silicon Optimization Layer 🔥

### High Priority (Week 2):
4. **Issue #18**: Real-Time Streaming Voice Pipeline (Updated) 🔥
5. **Issue #5**: Refactor Large Files (Updated) 🔧
6. **NEW**: Voice Cloning System 🎤

### Medium Priority (Week 3):
7. **Issue #8**: Implement LRU Cache for Vector Embeddings 🔧
8. **Issue #22**: Performance Monitoring Dashboard (Merge with NEW) 📊
9. **Issue #17**: Add Integration Test Coverage 🧪

### Low Priority (Backlog):
- Issue #11: Add OpenAPI Schema Customization
- Issue #12: Add Dependency Injection Pattern
- Issue #13: Consolidate Configuration Management
- Issue #14: Add Comprehensive Type Hints
- Issue #16: Improve Error Messages with Context

### Future Enhancements (After Streaming):
- Issue #19: Multi-User Support with Isolation
- Issue #20: Knowledge Graph for Advanced Memory
- Issue #21: Advanced Tool Ecosystem
- Issue #23: Automatic Conversation Summarization

---

## GitHub CLI Commands to Execute

```bash
# Close completed/irrelevant issues
gh issue close 15 -c "Superseded by streaming refactor"
gh issue close 10 -c "Being redesigned for streaming WebSocket"

# Update existing issues
gh issue edit 18 --title "Implement XTTS v2 Streaming Pipeline with M3 Optimization" --body-file issue_18_update.md
gh issue edit 5 --title "Modularize Core Components for Streaming Architecture" --body-file issue_5_update.md
gh issue edit 6 --title "Implement Transaction Management for Streaming Operations" --body-file issue_6_update.md

# Create new issues
gh issue create --title "Add M3 Max Hardware Acceleration Support" --body-file new_issue_m3.md --label "enhancement,performance,apple-silicon"
gh issue create --title "Implement Voice Cloning with XTTS v2" --body-file new_issue_voice_cloning.md --label "feature,audio,xtts"
gh issue create --title "Real-Time Performance Monitoring Dashboard" --body-file new_issue_monitoring.md --label "monitoring,ui,performance"

# Add milestones
gh api repos/:owner/:repo/milestones -f title="Phase 1: Foundation" -f due_on="2025-11-05"
gh api repos/:owner/:repo/milestones -f title="Phase 2: Streaming" -f due_on="2025-11-12"
gh api repos/:owner/:repo/milestones -f title="Phase 3: Optimization" -f due_on="2025-11-19"

# Assign issues to milestones
gh issue edit 6 7 --milestone "Phase 1: Foundation"
gh issue edit 5 18 --milestone "Phase 2: Streaming"
gh issue edit 8 22 --milestone "Phase 3: Optimization"
```

---

## Issue Tracking Dashboard

### Week 1 Sprint (Foundation):
- [ ] Fix HTTP Client Resource Leaks (#7)
- [ ] Database Transaction Management (#6)
- [ ] Apple Silicon Optimization (NEW)

### Week 2 Sprint (Streaming):
- [ ] XTTS v2 Streaming Pipeline (#18)
- [ ] Refactor Large Files (#5)
- [ ] Voice Cloning System (NEW)

### Week 3 Sprint (Polish):
- [ ] LRU Cache Implementation (#8)
- [ ] Performance Dashboard (#22 + NEW)
- [ ] Integration Tests (#17)

---

## Success Criteria

### By End of Week 1:
- ✅ All critical bugs fixed
- ✅ M3 optimizations in place
- ✅ Basic pipeline operational

### By End of Week 2:
- ✅ Streaming pipeline working
- ✅ <300ms latency achieved
- ✅ Voice cloning functional

### By End of Week 3:
- ✅ <200ms latency optimized
- ✅ Full monitoring dashboard
- ✅ 90% test coverage

---

## Notes for Implementation

1. **Always test on M3 Max** - Performance characteristics differ from x86/CUDA
2. **Use feature flags** - Allow gradual rollout of streaming features
3. **Document M3-specific setup** - Many developers won't have Apple Silicon experience
4. **Monitor thermals** - Though unlikely issue with M3 Max
5. **Test battery performance** - Ensure efficiency on battery power

---

## Communication Template for Team

```markdown
Subject: AI Assistant Roadmap Update - Streaming Implementation with M3 Optimization

Team,

We're restructuring our roadmap to implement real-time streaming with XTTS v2, optimized for Apple Silicon M3 Max hardware. This will achieve <200ms voice-to-voice latency.

Key Changes:
- Closing 6 completed/obsolete issues
- Updating 3 major issues with streaming scope
- Creating 3 new issues for M3 optimization and voice cloning
- Prioritizing into 3 weekly sprints

Please review the updated issues and claim any you'd like to work on.

Target: Fully operational streaming assistant by end of Week 3.
```