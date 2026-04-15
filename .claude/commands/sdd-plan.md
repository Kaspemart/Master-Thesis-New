# /sdd-plan — Create Technical Plan

Design the implementation approach for an approved specification.






---

## PREREQUISITES

Before proceeding, verify:
1. An approved spec exists: `.specs/feat-{name}/spec.md` with `Status: Approved`
2. If spec is not approved, inform user: "Please approve the specification first, or run /sdd-spec to continue refining it."

---

## PROCESS

### Step 1: Load Context

1. **Read the specification** (`.specs/feat-{name}/spec.md`)
   - Understand what needs to be built
   - Note acceptance criteria
   - Check exceptions and constraints

2. **Read the constitution** (`CLAUDE.md`)
   - Know project standards
   - Understand architectural patterns
   - Note approved libraries/technologies

3. **Scan the codebase**
   - Identify affected components
   - Find existing patterns to follow
   - Locate related code

4. **Check reference materials** (`.specs/feat-{name}/refs/`)
   - Review any technical documentation
   - Check diagrams or mockups

### Step 2: Analyze Impact

Present analysis to user:

```
📊 IMPLEMENTATION ANALYSIS

Specification: feat-{name}
Status: Approved

AFFECTED COMPONENTS:
  • src/services/user_service.py — Needs modification (add auth methods)
  • src/models/user.py — Needs modification (add token fields)
  • src/api/routes/auth.py — New file
  • src/api/middleware/auth.py — New file

DEPENDENCIES:
  • PyJWT (new dependency needed)
  • Existing UserService (will extend)

INTEGRATION POINTS:
  • Database: users table needs migration
  • API: New /auth/* endpoints
  • Middleware: Request authentication

ESTIMATED SCOPE:
  • New files: 3
  • Modified files: 2
  • New tests: ~10
  • Database migrations: 1
```

### Step 3: Design Architecture

Create the technical design:

```
PROPOSED ARCHITECTURE

┌─────────────────────────────────────────────────────────────┐
│                       API Layer                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ /auth/login │  │/auth/refresh│  │ /auth/logout│         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                │                │                 │
│         └────────────────┼────────────────┘                 │
│                          ▼                                  │
│              ┌───────────────────────┐                      │
│              │    AuthMiddleware     │                      │
│              └───────────┬───────────┘                      │
└──────────────────────────┼──────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                     Service Layer                            │
│              ┌───────────────────────┐                       │
│              │     AuthService       │                       │
│              │  • login()            │                       │
│              │  • refresh_token()    │                       │
│              │  • validate_token()   │                       │
│              └───────────┬───────────┘                       │
│                          │                                   │
│              ┌───────────┴───────────┐                       │
│              │     UserService       │                       │
│              │  (existing, extended) │                       │
│              └───────────────────────┘                       │
└──────────────────────────────────────────────────────────────┘
```

### Step 4: Define API Contracts

If the feature involves APIs:

```
API CONTRACTS

POST /auth/login
  Request:
    {
      "email": string,
      "password": string
    }
  Response (200):
    {
      "access_token": string,
      "refresh_token": string,
      "expires_in": number
    }
  Errors:
    401 - Invalid credentials
    422 - Validation error

POST /auth/refresh
  Request:
    {
      "refresh_token": string
    }
  Response (200):
    {
      "access_token": string,
      "expires_in": number
    }
  Errors:
    401 - Invalid/expired token
```

### Step 5: Document Database Changes

If applicable:

```
DATABASE CHANGES

Migration: 001_add_auth_tokens

ALTER TABLE users:
  + password_hash VARCHAR(255) NOT NULL
  + last_login TIMESTAMP NULL
  + refresh_token VARCHAR(255) NULL
  + token_expires_at TIMESTAMP NULL

New indexes:
  + idx_users_refresh_token ON users(refresh_token)
```

### Step 6: Identify Risks

```
RISKS AND MITIGATIONS

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Token leakage | Medium | High | Use secure cookies, short expiry |
| Brute force attacks | High | Medium | Rate limiting, account lockout |
| Session fixation | Low | High | Regenerate tokens on login |
```

### Step 7: Generate plan.md

Create `.specs/feat-{name}/plan.md`:

```markdown
# Technical Plan: {Feature Name}

**Status:** Draft
**Spec:** .specs/feat-{name}/spec.md
**Created:** {date}
**Last Updated:** {date}

## 1. Overview
{Brief description of technical approach}

## 2. Architecture
{ASCII diagram from Step 3}

### 2.1 Component Overview
{Description of components}

### 2.2 Affected Components
| Component | Change Type | Description |
|-----------|-------------|-------------|
| ... | ... | ... |

## 3. API Design
{From Step 4}

## 4. Database Changes
{From Step 5}

## 5. Dependencies
### 5.1 New Dependencies
| Package | Version | Justification |
|---------|---------|---------------|
| ... | ... | ... |

### 5.2 Internal Dependencies
| Component | Dependency Type | Notes |
|-----------|-----------------|-------|
| ... | ... | ... |

## 6. Security Considerations
{From spec constraints + best practices}

## 7. Testing Strategy
| Test Type | Scope | Approach |
|-----------|-------|----------|
| Unit | AuthService | Mock dependencies |
| Integration | Auth flow | Test DB, real requests |
| Security | Auth endpoints | OWASP testing |

## 8. Risks and Mitigations
{From Step 6}

## 9. Implementation Notes
{Any additional guidance for implementation}
```

### Step 8: Present for Approval

```
📋 TECHNICAL PLAN COMPLETE

I've created .specs/feat-{name}/plan.md

Key decisions:
  • JWT-based authentication with refresh tokens
  • Extend existing UserService
  • Add AuthMiddleware for protected routes

New dependencies:
  • PyJWT ^2.8.0

Database changes:
  • 1 migration to users table

What would you like to do?
  1. Review the full plan (I'll display it)
  2. Mark as Approved and proceed to /sdd-tasks
  3. Discuss changes to the approach
  4. Save and continue later
```

---

## KEY BEHAVIORS

1. **Always verify spec is approved** — Don't plan unfinished specs
2. **Ground in existing code** — Reference actual files and patterns
3. **Be specific** — File names, function signatures, not vague descriptions
4. **Consider testability** — Every component should be testable
5. **Identify risks early** — Better to surface concerns now
6. **Match project conventions** — Follow existing patterns from CLAUDE.md
