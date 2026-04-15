# /sdd-tasks — Generate Task Breakdown

<!-- Implements: AC-IMPLEM-001, AC-IMPLEM-002 -->

Break an approved plan into atomic, ordered implementation tasks.






---

## PREREQUISITES

Before proceeding, verify:
1. An approved plan exists: `.specs/feat-{name}/plan.md` with `Status: Approved`
2. The spec is also approved
3. If not approved, inform user and offer to run /sdd-plan

---

## PROCESS

### Step 1: Load Context

1. **Read the plan** (`.specs/feat-{name}/plan.md`)
2. **Read the spec** (`.specs/feat-{name}/spec.md`)
3. **Identify components** from the plan's architecture section

### Step 2: Generate Tasks

Break the plan into atomic tasks following TDD pattern:

```
TASK GENERATION PRINCIPLES

1. Each task should be completable in one session
2. Tasks should have clear acceptance criteria
3. Follow TDD: tests first, then implementation
4. Order by dependencies (can't use what doesn't exist)
5. Mark parallelizable tasks with [P]
6. Each task ends with a passing test or verification
```

### Step 3: Create tasks.md

Generate `.specs/feat-{name}/tasks.md`:

```markdown
# Implementation Tasks: {Feature Name}

**Status:** Draft
**Plan:** .specs/feat-{name}/plan.md
**Created:** {date}
**Last Updated:** {date}

---

## Task Legend

- `[ ]` — Not started
- `[~]` — In progress
- `[x]` — Complete
- `[P]` — Can be parallelized with adjacent [P] tasks
- `[B]` — Blocked (note reason)

---

## Phase 1: Setup & Foundation

### Task 1.1: Create feature branch
- [ ] Create branch `feat/{feature-name}`
- [ ] Verify spec and plan are accessible
- Acceptance: Branch created, CI passes

### Task 1.2: Add dependencies
- [ ] Add PyJWT to pyproject.toml
- [ ] Run `uv sync` or equivalent
- [ ] Verify import works
- Acceptance: `import jwt` succeeds

### Task 1.3: Create database migration
- [ ] Create migration file `001_add_auth_fields.py`
- [ ] Add password_hash, refresh_token, token_expires_at to users
- [ ] Add index on refresh_token
- [ ] Run migration
- Acceptance: Migration applies cleanly, rollback works

---

## Phase 2: Core Implementation (TDD)

### Task 2.1: AuthService - Token Generation [P]
**Tests first:**
- [ ] Write test: `test_generate_access_token_returns_valid_jwt`
- [ ] Write test: `test_access_token_contains_user_id`
- [ ] Write test: `test_access_token_expires_correctly`

**Implementation:**
- [ ] Create `src/services/auth_service.py`
- [ ] Implement `generate_access_token(user_id: int) -> str`
- [ ] Implement `generate_refresh_token(user_id: int) -> str`

**Verify:**
- [ ] All tests pass
- [ ] Linting passes
- Acceptance: `pytest tests/services/test_auth_service.py` passes

### Task 2.2: AuthService - Token Validation [P]
**Tests first:**
- [ ] Write test: `test_validate_token_returns_user_id`
- [ ] Write test: `test_validate_token_rejects_expired`
- [ ] Write test: `test_validate_token_rejects_invalid`

**Implementation:**
- [ ] Implement `validate_access_token(token: str) -> int | None`
- [ ] Implement `validate_refresh_token(token: str) -> int | None`

**Verify:**
- [ ] All tests pass
- Acceptance: Token validation works for valid/invalid/expired tokens

### Task 2.3: AuthService - Login Flow
**Tests first:**
- [ ] Write test: `test_login_success_returns_tokens`
- [ ] Write test: `test_login_invalid_password_fails`
- [ ] Write test: `test_login_unknown_user_fails`

**Implementation:**
- [ ] Implement `login(email: str, password: str) -> TokenPair | None`
- [ ] Add password hashing utility
- [ ] Integrate with UserService

**Verify:**
- [ ] All tests pass
- Acceptance: Login flow works end-to-end in tests

---

## Phase 3: API Layer

### Task 3.1: Auth Routes
**Tests first:**
- [ ] Write test: `test_login_endpoint_returns_tokens`
- [ ] Write test: `test_login_endpoint_validates_input`
- [ ] Write test: `test_refresh_endpoint_returns_new_token`

**Implementation:**
- [ ] Create `src/api/routes/auth.py`
- [ ] Implement POST /auth/login
- [ ] Implement POST /auth/refresh
- [ ] Register routes in main app

**Verify:**
- [ ] All tests pass
- [ ] Manual test with curl/httpie
- Acceptance: API endpoints respond correctly

### Task 3.2: Auth Middleware
**Tests first:**
- [ ] Write test: `test_middleware_allows_valid_token`
- [ ] Write test: `test_middleware_rejects_missing_token`
- [ ] Write test: `test_middleware_rejects_expired_token`

**Implementation:**
- [ ] Create `src/api/middleware/auth.py`
- [ ] Implement token extraction from header
- [ ] Implement validation and user injection
- [ ] Apply to protected routes

**Verify:**
- [ ] All tests pass
- Acceptance: Protected routes require valid auth

---

## Phase 4: Integration & Polish

### Task 4.1: Integration Tests
- [ ] Write test: Full login → access protected route → refresh → access again
- [ ] Write test: Token expiry and refresh flow
- [ ] Write test: Invalid token handling
- Acceptance: All integration tests pass

### Task 4.2: Documentation
- [ ] Add docstrings to all public functions
- [ ] Update API documentation (if exists)
- [ ] Add usage examples to README (if needed)
- Acceptance: Code is self-documenting

### Task 4.3: Final Verification
- [ ] Run full test suite: `pytest`
- [ ] Run linting: `ruff check`
- [ ] Run type checking (if configured)
- [ ] Manual smoke test of full flow
- Acceptance: All checks pass, feature works

---

## Phase 5: Completion

### Task 5.1: Pull Request
- [ ] Create PR with description referencing spec
- [ ] Ensure CI passes
- [ ] Request review
- Acceptance: PR created, ready for review

### Task 5.2: Address Feedback
- [ ] Address review comments
- [ ] Update tests if needed
- [ ] Get approval
- Acceptance: PR approved

### Task 5.3: Merge & Deploy
- [ ] Merge to main
- [ ] Verify deployment (if applicable)
- [ ] Update spec status to Completed
- Acceptance: Feature live, spec marked complete

---

## Commit Convention

All commits for this feature should follow:
```
type(auth): description

Refs: .specs/feat-{name}/spec.md
```

Example:
```
feat(auth): add JWT token generation

- Implement access and refresh token generation
- Add token expiry configuration
- Include user_id in token payload

Refs: .specs/feat-user-auth/spec.md
```

---

## Notes

{Space for implementation notes during execution}
```

### Step 4: Present for Approval

```
📋 TASK BREAKDOWN COMPLETE

I've created .specs/feat-{name}/tasks.md

Summary:
  • 5 phases, 13 tasks total
  • Estimated: {rough scope based on tasks}
  • Parallelizable: Tasks 2.1 and 2.2 can run in parallel

Task order:
  1. Setup (branch, dependencies, migration)
  2. Core TDD (AuthService)
  3. API Layer (routes, middleware)
  4. Integration & polish
  5. PR and completion

What would you like to do?
  1. Review the full task list
  2. Mark as Approved and start with /sdd-implement
  3. Adjust task breakdown
  4. Save for later
```

---

## KEY BEHAVIORS

1. **Atomic tasks** — Each task should be doable in one sitting
2. **TDD by default** — Tests before implementation
3. **Clear acceptance** — Each task has verification criteria
4. **Respect dependencies** — Order matters
5. **Mark parallelizable** — Help with planning
6. **Include cleanup** — Documentation, PR, deployment
