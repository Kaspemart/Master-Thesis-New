# /sdd-implement — Execute Implementation Tasks

<!-- Implements: AC-SDDVER-008, AC-IMPLEM-005, AC-IMPLEM-006, AC-IMPLEM-008 -->

Execute tasks one at a time following TDD methodology.






---

## PREREQUISITES

Before proceeding, verify:
1. Approved tasks exist: `.specs/feat-{name}/tasks.md` with `Status: Approved`
2. The plan and spec are also approved
3. Feature branch exists (or create it)

---

## PROCESS

### Step 1: Load Context

1. **Read tasks.md** — Know what needs to be done
2. **Read plan.md** — Understand the architecture
3. **Read spec.md** — Know the acceptance criteria
4. **Check current progress** — Find incomplete tasks

### Step 2: Find Next Task

Scan tasks.md for the next incomplete task:

```
📋 IMPLEMENTATION STATUS

Feature: feat-{name}
Branch: feat/{name}

Progress: 5/13 tasks complete (38%)

Completed:
  ✓ 1.1: Create feature branch
  ✓ 1.2: Add dependencies
  ✓ 1.3: Create database migration
  ✓ 2.1: AuthService - Token Generation
  ✓ 2.2: AuthService - Token Validation

Next task:
  → 2.3: AuthService - Login Flow

Ready to start Task 2.3?
```

### Step 3: Execute Task (TDD Cycle)

For each task, follow the TDD cycle:

#### 3.1 RED Phase — Write Failing Tests

```
🔴 RED PHASE: Writing Tests

Task 2.3: AuthService - Login Flow

I'll write the tests first:

tests/services/test_auth_service.py:
```python
class TestAuthServiceLogin:
    def test_login_success_returns_tokens(self, auth_service, test_user):
        """Successful login returns access and refresh tokens."""
        result = auth_service.login("test@example.com", "correct_password")

        assert result is not None
        assert "access_token" in result
        assert "refresh_token" in result

    def test_login_invalid_password_fails(self, auth_service, test_user):
        """Login with wrong password returns None."""
        result = auth_service.login("test@example.com", "wrong_password")

        assert result is None

    def test_login_unknown_user_fails(self, auth_service):
        """Login with unknown email returns None."""
        result = auth_service.login("nobody@example.com", "any_password")

        assert result is None
```

Running tests to confirm they fail...
```
$ pytest tests/services/test_auth_service.py::TestAuthServiceLogin -v
FAILED test_login_success_returns_tokens - AttributeError: 'AuthService' has no attribute 'login'
```

Tests fail as expected. Moving to GREEN phase.
```

#### 3.2 GREEN Phase — Implement

```
🟢 GREEN PHASE: Implementation

Now implementing the login method:

src/services/auth_service.py:
```python
from passlib.hash import bcrypt

class AuthService:
    # ... existing methods ...

    def login(self, email: str, password: str) -> dict | None:
        """
        Authenticate user and return tokens.

        Args:
            email: User's email address
            password: Plain text password

        Returns:
            Dict with access_token and refresh_token, or None if auth fails
        """
        user = self.user_service.get_by_email(email)
        if not user:
            return None

        if not bcrypt.verify(password, user.password_hash):
            return None

        return {
            "access_token": self.generate_access_token(user.id),
            "refresh_token": self.generate_refresh_token(user.id),
        }
```

Running tests...
```
$ pytest tests/services/test_auth_service.py::TestAuthServiceLogin -v
PASSED test_login_success_returns_tokens
PASSED test_login_invalid_password_fails
PASSED test_login_unknown_user_fails
```

All tests pass!
```

#### 3.3 REFACTOR Phase (if needed)

```
🔵 REFACTOR PHASE: Clean Up

Checking for improvements:
  • Code is clean and readable ✓
  • No duplication ✓
  • Type hints present ✓
  • Docstrings added ✓

Running full test suite to ensure no regressions...
```
$ pytest
====== 15 passed in 0.8s ======
```

Linting check...
```
$ ruff check src/services/auth_service.py
All checks passed!
```

Task complete!
```

### Step 4: Update Progress

Update tasks.md:
- Mark completed tasks with `[x]`
- Update any notes
- Mark current task as `[~]` if in progress

```markdown
### Task 2.3: AuthService - Login Flow
**Tests first:**
- [x] Write test: `test_login_success_returns_tokens`
- [x] Write test: `test_login_invalid_password_fails`
- [x] Write test: `test_login_unknown_user_fails`

**Implementation:**
- [x] Implement `login(email: str, password: str) -> TokenPair | None`
- [x] Add password hashing utility
- [x] Integrate with UserService

**Verify:**
- [x] All tests pass
- Acceptance: Login flow works end-to-end in tests ✓
```

### Step 5: Commit

```
📝 COMMITTING CHANGES

Commit message:
```
feat(auth): implement login flow with password verification

- Add login method to AuthService
- Integrate with UserService for user lookup
- Use bcrypt for password verification
- Return JWT tokens on successful auth

Refs: .specs/feat-user-auth/spec.md

Co-Authored-By: sdd
```

Committed: abc1234

Files changed:
  • src/services/auth_service.py (modified)
  • tests/services/test_auth_service.py (modified)
```

### Step 6: Continue or Pause

```
✅ TASK 2.3 COMPLETE

Progress: 6/13 tasks (46%)

Next task: 3.1 - Auth Routes

Would you like to:
  1. Continue to next task (3.1: Auth Routes)
  2. Take a break (progress is saved)
  3. Review what we've built so far
  4. Run full test suite before continuing
```

---

## HANDLING ISSUES

### When Tests Fail

```
⚠️ TEST FAILURE

Test: test_login_invalid_password_fails
Error: AssertionError: expected None, got {'access_token': ...}

This suggests the password verification isn't working correctly.

Options:
  1. Debug together — I'll help investigate
  2. Show me the relevant code — I'll review
  3. Skip for now — Mark as blocked and continue
```

### When Blocked

```
🚫 BLOCKED

Task 3.2 (Auth Middleware) is blocked:
  Reason: Need to decide on header format (Authorization vs X-Auth-Token)

This should have been decided in the spec. Options:
  1. Make a decision now (I'll recommend)
  2. Update the spec with this decision
  3. Skip and mark as blocked
  4. Ask the team/stakeholder
```

### When Deviating from Plan

```
⚠️ DEVIATION DETECTED

The plan specified using middleware, but you're implementing
authentication as a dependency injection.

This is a valid alternative approach. Should I:
  1. Continue with your approach (update plan to match)
  2. Revert to the planned approach
  3. Discuss trade-offs before deciding
```

---

## KEY BEHAVIORS

1. **One task at a time** — Focus, complete, move on
2. **TDD strictly** — Red → Green → Refactor
3. **Commit frequently** — Each task = one commit minimum
4. **Reference the spec** — Every commit links back
5. **Stop when blocked** — Don't hack around issues
6. **Update tasks.md** — Keep progress visible
7. **Run tests often** — Catch regressions early

---

## COMPLETION

When all tasks are done, **run verification before marking complete**:

### Step 7: Run Verification

```
🔍 VERIFICATION PHASE

Running sdd verify to validate implementation...

$ sdd verify --feature feat-{name}
```

**If verification passes:**
```
╭─────────────────────── Verification: feat-{name} ────────────────────────╮
│ Status: PASSED                                                            │
│ Coverage: 100% (5/5 ACs)                                                  │
│ Tests: passed (2.3s)                                                      │
╰───────────────────────────────────────────────────────────────────────────╯

✓ Spec status updated to 'Verified'

🎉 IMPLEMENTATION COMPLETE

Feature: feat-{name}
All tasks completed and verified!

Summary:
  • All acceptance criteria covered
  • All tests passing
  • Spec status: Verified

Next steps:
  1. Run 'sdd pr-check' to confirm PR readiness
  2. Create pull request with 'glab mr create'
  3. Celebrate! 🎉

Run /sdd-status to see updated project overview.
```

**If verification fails:**
```
╭─────────────────────── Verification: feat-{name} ────────────────────────╮
│ Status: FAILED - Missing AC coverage                                      │
│ Coverage: 80% (4/5 ACs)                                                   │
╰───────────────────────────────────────────────────────────────────────────╯

⚠️ VERIFICATION FAILED

Do NOT mark implementation as complete until verification passes.

Missing coverage:
  • AC-XXX-003: Missing test reference

Actions:
  1. Add test reference: # Test for AC-XXX-003
  2. Re-run verification: sdd verify --feature feat-{name}
  3. Continue until all ACs are covered
```

---

### PR Creation Blocking

Before creating a PR, verification must pass:

```bash
# Check if ready for PR
sdd pr-check --feature feat-{name}

# If ready, create PR
glab mr create --title "feat: description" --description "..."
```

If `sdd pr-check` fails, it will show what needs to be fixed before the PR can be created.
