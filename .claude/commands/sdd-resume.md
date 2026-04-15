# /sdd-resume — Continue Incomplete Work

Resume an incomplete specification, plan, or implementation from where you left off.




---

## PROCESS

### Step 1: Find Incomplete Work

Scan `.specs/` for incomplete work:
- Specs with Status != "Approved" and != "Completed"
- Plans with Status != "Approved"
- Tasks with incomplete checkboxes

### Step 2: Present Options

```
╔══════════════════════════════════════════════════════════════════╗
║                    Resume Incomplete Work                         ║
╚══════════════════════════════════════════════════════════════════╝

Found 3 items that can be resumed:

────────────────────────────────────────────────────────────────────

1. feat-user-auth/spec.md

   Phase: Requirements Gathering
   Progress: 60%
   Last session: 2 hours ago

   Outstanding:
     • [ ] Confirm OAuth providers (Google, GitHub, or both?)
     • [ ] Clarify session timeout requirements
     • [ ] Decision needed: MFA in v1 or defer?

   Session notes available: Yes

────────────────────────────────────────────────────────────────────

2. feat-dark-mode/spec.md

   Phase: Initial Interview
   Progress: 30%
   Last session: 3 days ago

   Outstanding:
     • [ ] Theme persistence mechanism
     • [ ] System preference detection
     • [ ] Color palette decisions

   Session notes available: Yes

────────────────────────────────────────────────────────────────────

3. feat-api-v2/tasks.md

   Phase: Implementation
   Progress: 4/12 tasks (33%)
   Last session: 1 day ago

   Next task: 2.3 - Endpoint validation

   Session notes available: Yes

────────────────────────────────────────────────────────────────────

Which would you like to continue? [1/2/3]
```

### Step 3: Load Context

After user selects, load full context:

```
📂 LOADING CONTEXT: feat-user-auth

Reading session notes...
Reading specification...
Scanning for changes since last session...

Done!
```

### Step 4: Summarize Reference Materials

```
📁 REFERENCE MATERIALS (4 files in refs/):

  • api-docs.md (12KB)
    OAuth provider API documentation, covers Google and GitHub
    integration patterns

  • existing-flow.png (45KB)
    Screenshot of current login screen

  • security-requirements.txt (2KB)
    Compliance requirements from legal team
    ⭐ NEW — Added since your last session

  • competitor-analysis.md (8KB)
    Analysis of how competitors handle auth

💡 SUGGESTED REVIEW:
   security-requirements.txt was added since your last session.
   This may affect the MFA decision we left open.

   Would you like me to summarize it? [Y/n]
```

### Step 5: Detect Project Changes

```
📊 PROJECT CHANGES SINCE LAST SESSION:

Git activity (last 2 hours):
  • 3 commits to src/auth/ by @teammate
  • New file: src/services/token_service.py
  • Modified: src/models/user.py (added email_verified field)

💡 RELEVANCE CHECK:
   The new token_service.py might be relevant to your auth
   implementation. It appears to handle token generation already.

   Should we review this before continuing? [Y/n]
```

### Step 6: Context Summary

```
📝 WHERE WE LEFT OFF:

Last Session Summary:
  You were working on the user authentication specification.
  We completed the basic understanding (Layer 1) and implications
  (Layer 2) of the interview.

Key Decisions Made:
  • JWT-based authentication (not session-based)
  • Separate access and refresh tokens
  • 15-minute access token expiry
  • 7-day refresh token expiry

Outstanding Questions:
  1. OAuth providers: Google only, or also GitHub?
     → You wanted to check with product team

  2. Session timeout: How long before requiring re-auth?
     → Depends on security requirements (now available in refs!)

  3. MFA: Include in v1 or defer to v2?
     → Wanted to assess complexity first

Discussion Notes:
  "User mentioned legacy system integration may affect our
  approach. The existing login system uses session cookies,
  so we need a migration path for existing users."
```

### Step 7: Resume

```
────────────────────────────────────────────────────────────────────

Ready to continue?

Based on the new security requirements file, I'd suggest we:
1. Review the security requirements first
2. Then address the MFA question (likely answered there)
3. Continue with Layer 3 (Best Practices) of the interview

Alternatively:
  • Jump to a specific outstanding question
  • Start fresh on a different section
  • Review what we have so far

What would you like to do?
```

---

## RESUMING IMPLEMENTATION

If resuming tasks.md:

```
📋 RESUMING IMPLEMENTATION: feat-user-auth

Progress: 6/13 tasks (46%)

Last completed: Task 2.3 - AuthService Login Flow
  Committed: abc1234 (2 hours ago)

Next task: 3.1 - Auth Routes
  Phase: API Layer
  Subtasks:
    • Write test: test_login_endpoint_returns_tokens
    • Write test: test_login_endpoint_validates_input
    • Write test: test_refresh_endpoint_returns_new_token
    • Create src/api/routes/auth.py
    • Implement POST /auth/login
    • Implement POST /auth/refresh
    • Register routes in main app

Current branch: feat/user-auth
Last commit: "feat(auth): implement login flow" (abc1234)

Tests status: All 15 tests passing

Ready to continue with Task 3.1? [Y/n]
```

---

## NO INCOMPLETE WORK

```
✅ NO INCOMPLETE WORK FOUND

All specifications are either:
  • Completed
  • Approved and ready for next phase

Current status:
  • feat-user-auth: Completed ✓
  • feat-export: Completed ✓
  • feat-api-v2: Approved (ready for /sdd-plan)

Would you like to:
  1. Start a new specification with /sdd-spec
  2. Continue feat-api-v2 with /sdd-plan
  3. View full status with /sdd-status
```

---

## KEY BEHAVIORS

1. **Restore full context** — User shouldn't need to remember anything
2. **Highlight changes** — New refs, project changes since last session
3. **Suggest next action** — Don't just dump info, guide the user
4. **Support non-linear work** — User can jump to any section
5. **Preserve decisions** — All previous choices remain valid
6. **Note relevance** — Connect new information to open questions
