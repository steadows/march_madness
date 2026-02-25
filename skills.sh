#!/bin/bash
# ============================================================
# SKILLS REGISTRY — Run this to see available agent skills
# Usage: bash skills.sh [skill-name]
#   No args    → list all skills with descriptions
#   skill-name → cat the full skill file
# ============================================================

SKILLS_DIR=".claude/skills"
COMMUNITY_DIR=".claude/skills/community"

if [ -z "$1" ]; then
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                   AVAILABLE AGENT SKILLS                       ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    echo "║                                                                ║"
    echo "║  CUSTOM SKILLS (project-specific)                              ║"
    echo "║  ─────────────────────────────────                             ║"
    echo "║  project-conventions    Code style, naming, file structure     ║"
    echo "║  data-engineering       Data loading, features, transforms     ║"
    echo "║  model-training         Model configs, training protocol       ║"
    echo "║  evaluation-validation  CV strategy, Brier, calibration        ║"
    echo "║  ensemble-submission    Ensembling, submission format           ║"
    echo "║  session-management     Context limits, checkpointing          ║"
    echo "║  debugging              Common errors, diagnosis patterns      ║"
    echo "║  hyperparameter-tuning  Optuna, search spaces, strategies      ║"
    echo "║  external-data          BartTorvik, KenPom, Vegas lines        ║"
    echo "║                                                                ║"
    echo "║  COMMUNITY SKILLS (from skills.sh)                             ║"
    echo "║  ─────────────────────────────────                             ║"
    echo "║  c/python-testing-patterns     Pytest fixtures, parametrize    ║"
    echo "║  c/python-performance-optimization  Profiling, fast pandas     ║"
    echo "║  c/python-design-patterns      Factory, strategy, clean arch   ║"
    echo "║  c/data-quality-frameworks     Validation suites, assertions   ║"
    echo "║  c/python-error-handling       Exception patterns, fallbacks   ║"
    echo "║  c/python-type-safety          Type hints, mypy patterns       ║"
    echo "║  c/test-driven-development     TDD workflow, test-first        ║"
    echo "║  c/testing-anti-patterns       What NOT to do in tests         ║"
    echo "║  c/git-commit                  Commit message conventions      ║"
    echo "║                                                                ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    echo "║  Usage:                                                        ║"
    echo "║    bash skills.sh data-engineering         (custom skill)      ║"
    echo "║    bash skills.sh c/python-testing-patterns (community skill)  ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Phase → Skill mapping:"
    echo "  Phase 0 (Setup)     → project-conventions, c/python-project-structure"
    echo "  Phase 1 (Data)      → data-engineering, c/data-quality-frameworks, c/python-testing-patterns"
    echo "  Phase 2 (Features)  → data-engineering, c/python-performance-optimization, c/python-type-safety"
    echo "  Phase 3 (Training)  → model-training, evaluation-validation, c/test-driven-development"
    echo "  Phase 4 (Ensemble)  → ensemble-submission, evaluation-validation"
    echo "  Phase 5 (Iteration) → hyperparameter-tuning, external-data"
    echo "  Debugging           → debugging, c/python-error-handling, c/testing-anti-patterns"
    echo "  Commits             → c/git-commit"
    echo ""
    echo "⚠️  CONTEXT WARNING: Load 1-2 skills max per session. Community skills are large."
    echo "    Custom skills: ~1K tokens each. Community skills: ~3-6K tokens each."
else
    # Handle c/ prefix for community skills
    if [[ "$1" == c/* ]]; then
        SKILL_NAME="${1#c/}"
        SKILL_FILE="${COMMUNITY_DIR}/${SKILL_NAME}.md"
    else
        SKILL_FILE="${SKILLS_DIR}/${1}.md"
    fi

    if [ -f "$SKILL_FILE" ]; then
        cat "$SKILL_FILE"
    else
        echo "ERROR: Skill '${1}' not found at ${SKILL_FILE}"
        echo "Run 'bash skills.sh' with no args to see available skills."
    fi
fi
