# Skill: Session Management

## When to Use
Load this skill if you feel the conversation getting long, if you're unsure how to manage context, or at the START of any session.

## The Golden Rule
**The file system is your memory. The conversation is your scratch pad. CLAUDE.md is the bridge.**

## Context Budget
You have roughly 200K tokens of context. Here's how it fills:

| What | Tokens | You Control? |
|------|--------|-------------|
| System prompt | ~5K | No |
| CLAUDE.md (auto-loaded) | ~1.5K | Keep it lean |
| Each skill file you load | ~1-1.5K | Load 1-2 max |
| PIPELINE_BUILD_GUIDE.md | ~4K | Read once per session |
| Each file you Read | ~0.5-3K each | Minimize reads |
| Each bash command output | ~0.1-5K each | Redirect long ones |
| Your own reasoning | ~0.5K per turn | Be concise |
| Code you write (shown in chat) | ~1-3K each | Write to disk, not chat |

**Budget per session: ~50 tool calls before things get tight.**

## Techniques to Stay Within Budget

### 1. Write-First, Don't Draft
```
BAD:  "Here's the code I'll write: [200 lines shown in chat] Now let me write it..."
GOOD: [Uses Write tool directly — code only exists on disk, not in conversation]
```

### 2. Quiet Testing
```
BAD:  pytest tests/ -v                    # Full output per test
GOOD: pytest tests/ -q                    # One line per test
BEST: pytest tests/ -q > artifacts/test_results.txt 2>&1 && tail -5 artifacts/test_results.txt
```

### 3. Redirect Verbose Commands
```
BAD:  python3 -c "df.describe()"          # Prints huge tables to conversation
GOOD: python3 script.py > artifacts/eda_output.txt 2>&1 && tail -20 artifacts/eda_output.txt
```

### 4. Surgical File Reads
```
BAD:  Read the entire 500-line src/feature_engineering.py
GOOD: Read src/feature_engineering.py lines 1-30 (just the function signatures)
GOOD: Grep for the specific function you need
```

### 5. Don't Re-Read What You Just Wrote
If you just used the Write tool to create a file, you KNOW what's in it. Don't Read it back.

## When to Checkpoint and End Session

**Checkpoint triggers (do all 4):**
1. You've made 30+ tool calls in this session
2. You've completed a sub-task within the current phase
3. Tests pass for what you've built so far
4. You can describe remaining work in 1-2 sentences

**Checkpoint actions:**
1. Run `pytest tests/ -q` to verify nothing is broken
2. Update CLAUDE.md:
   - Current Phase status line
   - Any Key Decisions or Known Issues
   - Brier Scores if applicable
3. Commit: `git add src/ tests/ CLAUDE.md && git commit -m "checkpoint: <what's done>"`
4. Tell user: "Completed [X]. Remaining: [Y]. Recommend new session to continue."

## Phase Splitting Guide
If a phase is too big for one session:

| Phase | Can Split Into | Split Point |
|-------|---------------|-------------|
| Phase 1 | 1a: data_loader.py + tests | 1b: EDA notebook |
| Phase 2 | 2a: elo.py + massey.py | 2b: feature_engineering.py full pipeline |
| Phase 3 | 3a: CV framework + logistic baseline | 3b: GBMs + calibration |
| Phase 4 | Usually fits in one session | |

## Recovery: Starting a New Session Mid-Phase
When you start a new session and CLAUDE.md says a phase is in progress:
1. Read CLAUDE.md (auto-loaded)
2. Run `pytest tests/ -q` to see what's passing
3. Run `ls src/` to see what modules exist
4. Read ONLY the module you need to continue (not everything)
5. Continue from where the previous session left off
