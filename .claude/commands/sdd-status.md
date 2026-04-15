# /sdd-status — Show Specification Progress

Display the status of all specifications in the project.




---

## PROCESS

### Step 1: Scan .specs/

Look for all feature directories in `.specs/`:
- Each `feat-*` directory is a feature
- Read `spec.md`, `plan.md`, `tasks.md` from each

### Step 2: Parse Status

For each feature, extract:
- **Status** from spec.md header (Draft, In Progress, Approved, Completed)
- **Phase** (Spec, Plan, Tasks, Implement)
- **Progress** percentage
- **Last Updated** date

Determine phase:
- If no spec.md or spec not approved → Phase: Spec
- If spec approved but no plan.md → Phase: Plan
- If plan approved but no tasks.md → Phase: Tasks
- If tasks exist → Phase: Implement (calculate % from checkboxes)

### Step 3: Display Overview

```
╔══════════════════════════════════════════════════════════════════╗
║                    SDD Status Overview                            ║
╚══════════════════════════════════════════════════════════════════╝

PROJECT: {project_name}
CONSTITUTION: CLAUDE.md (last updated: 2025-12-20)

────────────────────────────────────────────────────────────────────

SPECIFICATIONS:

| Feature              | Phase     | Status      | Progress | Updated     |
|----------------------|-----------|-------------|----------|-------------|
| feat-user-auth       | Implement | In Progress | 46%      | 2 hours ago |
| feat-dark-mode       | Spec      | In Progress | 30%      | 3 days ago  |
| feat-api-v2          | Tasks     | Approved    | 0%       | 1 day ago   |
| feat-export          | Complete  | Completed   | 100%     | 1 week ago  |

────────────────────────────────────────────────────────────────────

SUMMARY:

  Total specifications: 4

  By status:
    • Completed:   1
    • In Progress: 2
    • Approved:    1
    • Draft:       0

  By phase:
    • Spec:       1
    • Plan:       0
    • Tasks:      1
    • Implement:  1
    • Complete:   1

────────────────────────────────────────────────────────────────────

STALE SPECIFICATIONS (no activity > 7 days):

  ⚠️  feat-dark-mode — Last updated 3 days ago (Spec phase, 30%)
      Consider: /sdd-resume to continue, or archive if abandoned

────────────────────────────────────────────────────────────────────

NEXT ACTIONS:

  feat-user-auth:
    → Continue implementation with /sdd-implement
    → 7 tasks remaining

  feat-api-v2:
    → Ready to start implementation with /sdd-implement
    → 12 tasks queued

  feat-dark-mode:
    → Resume specification with /sdd-resume
    → Outstanding questions need answers

────────────────────────────────────────────────────────────────────

EXCEPTIONS SUMMARY:

  Total exceptions: 3 across 2 specs

  ⚡ Pattern detected: "pandas instead of polars" (2 occurrences)
     Run /sdd-exceptions for details

────────────────────────────────────────────────────────────────────

Run /sdd-help for command reference
```

---

## DETAILED VIEW

If user asks for details on a specific feature:

```
📋 DETAILED STATUS: feat-user-auth

Specification: .specs/feat-user-auth/spec.md
  Status: Approved
  Created: 2025-12-20
  Author: John Developer

  Summary: JWT-based authentication for API access with
           refresh token support.

Plan: .specs/feat-user-auth/plan.md
  Status: Approved
  Architecture: AuthService + Middleware pattern
  Dependencies: PyJWT, passlib

Tasks: .specs/feat-user-auth/tasks.md
  Status: In Progress
  Progress: 6/13 (46%)

  Completed:
    ✓ 1.1: Create feature branch
    ✓ 1.2: Add dependencies
    ✓ 1.3: Create database migration
    ✓ 2.1: AuthService - Token Generation
    ✓ 2.2: AuthService - Token Validation
    ✓ 2.3: AuthService - Login Flow

  Remaining:
    • 3.1: Auth Routes
    • 3.2: Auth Middleware
    • 4.1: Integration Tests
    • 4.2: Documentation
    • 4.3: Final Verification
    • 5.1: Pull Request
    • 5.2: Address Feedback
    • 5.3: Merge & Deploy

Session: .specs/feat-user-auth/.session.md
  Last session: 2 hours ago
  Context saved: Yes

References: .specs/feat-user-auth/refs/
  • api-spec.md
  • security-requirements.txt

Exceptions: 1
  #1: Using bcrypt instead of argon2 (legacy compatibility)
```

---

## EMPTY STATE

If no specifications exist:

```
╔══════════════════════════════════════════════════════════════════╗
║                    SDD Status Overview                            ║
╚══════════════════════════════════════════════════════════════════╝

PROJECT: {project_name}

No specifications found.

Get started:
  1. Run /sdd-spec to create your first specification
  2. Follow the guided interview process
  3. Approve the spec when ready
  4. Continue through Plan → Tasks → Implement

Run /sdd-help for workflow overview.
```

---

## NOT INITIALIZED

If SDD is not set up:

```
⚠️  SDD NOT INITIALIZED

This project doesn't have SDD set up yet.

To initialize SDD:
  1. Run 'sdd init' in your terminal
  2. Review and customize CLAUDE.md
  3. Start specifying with /sdd-spec

For existing projects with code:
  Run 'sdd init --brown-field'
```
