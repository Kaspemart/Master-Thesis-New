# /sdd-verify — Verify Feature Implementation

<!-- Implements: AC-SDDVER-007, AC-SDDVER-008, AC-SDDVER-009 -->

Verify that a feature implementation satisfies its specification.

---

## PURPOSE

After implementing a feature with `/sdd-implement`, use this command to verify:
1. All acceptance criteria have implementation references
2. All acceptance criteria have test references
3. Tests pass

This closes the loop in the SDD workflow, ensuring the implementation actually matches the specification.

---

## PROCESS

### Step 1: Determine Feature to Verify

If a feature name is provided as argument, use it. Otherwise:
1. Check for active session in `.specs/feat-*/.session.md`
2. Check if cwd is within a `.specs/feat-*` directory
3. If neither, ask user to specify with `--feature`

### Step 2: Run Verification

Execute the verification using the CLI:

```bash
sdd verify --feature {feature-name}
```

Or if verifying all features:

```bash
sdd verify --all
```

### Step 3: Interpret Results

The command will output verification status:

**PASSED** — All acceptance criteria are covered and tests pass:
```
╭─────────────────────────────────────────────────────────────╮
│            Verification: feat-auth                          │
├─────────────────────────────────────────────────────────────┤
│ Status: PASSED                                              │
│ Coverage: 100% (5/5 ACs)                                    │
│ Tests: passed (2.3s)                                        │
╰─────────────────────────────────────────────────────────────╯

┌────────────────┬────────────────────┬──────┬──────┬─────────┐
│ AC ID          │ Title              │ Impl │ Test │ Status  │
├────────────────┼────────────────────┼──────┼──────┼─────────┤
│ AC-AUTH-001    │ Login works        │  ✓   │  ✓   │ Covered │
│ AC-AUTH-002    │ Token expires      │  ✓   │  ✓   │ Covered │
│ AC-AUTH-003    │ Refresh works      │  ✓   │  ✓   │ Covered │
│ AC-AUTH-004    │ Logout clears      │  ✓   │  ✓   │ Covered │
│ AC-AUTH-005    │ Invalid rejected   │  ✓   │  ✓   │ Covered │
└────────────────┴────────────────────┴──────┴──────┴─────────┘
```

**FAILED (Coverage)** — Missing AC references:
```
╭─────────────────────────────────────────────────────────────╮
│            Verification: feat-auth                          │
├─────────────────────────────────────────────────────────────┤
│ Status: FAILED - Missing AC coverage                        │
│ Coverage: 60% (3/5 ACs)                                     │
╰─────────────────────────────────────────────────────────────╯

┌────────────────┬────────────────────┬──────┬──────┬─────────┐
│ AC ID          │ Title              │ Impl │ Test │ Status  │
├────────────────┼────────────────────┼──────┼──────┼─────────┤
│ AC-AUTH-001    │ Login works        │  ✓   │  ✓   │ Covered │
│ AC-AUTH-002    │ Token expires      │  ✓   │  ✗   │ Partial │
│ AC-AUTH-003    │ Refresh works      │  ✗   │  ✗   │ Missing │
│ AC-AUTH-004    │ Logout clears      │  ✓   │  ✓   │ Covered │
│ AC-AUTH-005    │ Invalid rejected   │  ✗   │  ✗   │ Missing │
└────────────────┴────────────────────┴──────┴──────┴─────────┘

Tip: Reference missing ACs in your code:
     # Implements AC-AUTH-003
     def test_AC_AUTH_003_refresh_works():
```

**FAILED (Tests)** — Tests fail:
```
╭─────────────────────────────────────────────────────────────╮
│            Verification: feat-auth                          │
├─────────────────────────────────────────────────────────────┤
│ Status: FAILED - Tests failed                               │
│ Coverage: 100% (5/5 ACs)                                    │
│ Tests: failed (1.8s)                                        │
╰─────────────────────────────────────────────────────────────╯

Test failures indicate the implementation doesn't match the
specification. Fix the failing tests before verification can pass.
```

### Step 4: Handle Results

**If PASSED:**
```
✅ VERIFICATION PASSED

Feature: feat-auth
All 5 acceptance criteria are covered.
All tests pass.

Next steps:
  1. Update spec status to "Verified" or "Complete"
  2. Create pull/merge request
  3. Celebrate! 🎉
```

**If FAILED (Coverage):**
```
❌ VERIFICATION FAILED - Missing Coverage

To fix:
  1. Add implementation references for missing ACs:
     # Implements AC-AUTH-003

  2. Add test references for missing ACs:
     def test_AC_AUTH_003_refresh_works():

  3. Run verification again

Would you like me to help add the missing references?
```

**If FAILED (Tests):**
```
❌ VERIFICATION FAILED - Tests Failed

To fix:
  1. Review test output above
  2. Fix the failing tests
  3. Ensure implementation matches spec requirements
  4. Run verification again

Would you like me to help debug the test failures?
```

---

## OPTIONS

| Option | Description |
|--------|-------------|
| `--feature NAME` | Verify specific feature (e.g., `feat-auth`) |
| `--all` | Verify all features in the project |
| `--no-tests` | Skip test execution (coverage only) |
| `--format FORMAT` | Output format: `table`, `json`, `markdown` |
| `--verbose` | Show detailed test output |

---

## EXAMPLES

Verify current feature (auto-detected):
```bash
sdd verify
```

Verify specific feature:
```bash
sdd verify --feature feat-auth
```

Verify all features in project:
```bash
sdd verify --all
```

JSON output for CI/CD:
```bash
sdd verify --feature feat-auth --format json
```

Coverage check only (no tests):
```bash
sdd verify --feature feat-auth --no-tests
```

---

## AUTO-MINT

If the spec has unminted ACs (informal IDs like `AC-1:`, `AC-2:`), the verify command will automatically mint them before checking coverage, unless `--no-mint` is specified.

This ensures all ACs have traceable IDs like `AC-AUTH-001`.

---

## CI/CD INTEGRATION

Use the JSON output format for CI/CD pipelines:

```yaml
# GitLab CI example
verify:
  script:
    - sdd verify --all --format json > verification.json
  artifacts:
    reports:
      custom_report: verification.json
```

Exit codes:
- `0` — Verification passed
- `1` — Verification failed

---

## TROUBLESHOOTING

**"No feature specified and could not detect current feature"**
- Specify the feature: `sdd verify --feature feat-name`
- Or navigate to the feature's .specs directory

**"No minted acceptance criteria found"**
- Run `sdd mint .specs/feat-name/spec.md` first
- Or use `--auto-mint` to mint automatically

**Coverage shows 0% but I have references**
- Check that AC IDs match exactly (case-sensitive)
- Ensure references are in comments: `# Implements AC-XXX-001`

**Tests pass locally but fail in verification**
- Verification runs from project root
- Check working directory assumptions in tests
