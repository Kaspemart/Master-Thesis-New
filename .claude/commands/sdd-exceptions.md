# /sdd-exceptions — List All Exceptions

Display all exceptions across the project with pattern detection and recommendations.




---

## PROCESS

### Step 1: Scan All Specs

Search all `.specs/feat-*/spec.md` files for Exception tables.

### Step 2: Aggregate and Analyze

Parse exceptions from each spec:
```markdown
## 7. Exceptions

| ID | Standard | Deviation | Justification |
|----|----------|-----------|---------------|
| 1 | Use polars (CLAUDE.md §3.2) | Using pandas | Legacy API returns DataFrames |
```

### Step 3: Display Overview

```
╔══════════════════════════════════════════════════════════════════╗
║                    SDD Exceptions Overview                        ║
╚══════════════════════════════════════════════════════════════════╝

PROJECT: {project_name}
CONSTITUTION: CLAUDE.md (last updated: 2025-12-20)

────────────────────────────────────────────────────────────────────

EXCEPTIONS BY SPECIFICATION:

feat-data-import (2 exceptions):

  #1: Library Exception
      Standard:      Use polars for dataframes (CLAUDE.md §3.2)
      Deviation:     Using pandas
      Justification: Legacy API returns pandas DataFrames, conversion
                     overhead not justified for this use case
      Approved by:   self (John Developer)
      Date:          2025-12-18

  #2: Architecture Exception
      Standard:      Async-first for I/O (CLAUDE.md §4.1)
      Deviation:     Synchronous code
      Justification: Simple CLI script, async overhead not justified
      Approved by:   self (John Developer)
      Date:          2025-12-18

────────────────────────────────────────────────────────────────────

feat-export (1 exception):

  #1: Library Exception
      Standard:      Use httpx for HTTP (CLAUDE.md §3.5)
      Deviation:     Using requests
      Justification: Third-party SDK requires requests as dependency
      Approved by:   self (Jane Developer)
      Date:          2025-12-15

────────────────────────────────────────────────────────────────────

feat-legacy-connector (1 exception):

  #1: Library Exception
      Standard:      Use polars for dataframes (CLAUDE.md §3.2)
      Deviation:     Using pandas
      Justification: Must match legacy system's data format exactly
      Approved by:   self (John Developer)
      Date:          2025-12-20

════════════════════════════════════════════════════════════════════

SUMMARY:

  Total exceptions: 4 across 3 specifications

  By type:
    • Library exceptions:     3
    • Architecture exceptions: 1

  By standard:
    • CLAUDE.md §3.2 (polars): 2 exceptions
    • CLAUDE.md §3.5 (httpx):  1 exception
    • CLAUDE.md §4.1 (async):  1 exception

════════════════════════════════════════════════════════════════════

⚡ PATTERNS DETECTED:

  PATTERN 1: "pandas instead of polars"
  ─────────────────────────────────────
  Occurrences: 2 (feat-data-import, feat-legacy-connector)
  Common reason: Legacy system compatibility

  💡 RECOMMENDATION:
     This exception appears multiple times. Consider updating
     CLAUDE.md §3.2 to explicitly allow pandas for legacy
     integrations:

     "Use polars for dataframes, except when interfacing with
      legacy systems that require pandas compatibility."

     This would make future specs cleaner and acknowledge
     the legitimate use case.

════════════════════════════════════════════════════════════════════

⚠️  POTENTIALLY OBSOLETE EXCEPTIONS:

  None found.

  (Obsolete exceptions occur when the constitution is updated
   to allow what was previously an exception)

════════════════════════════════════════════════════════════════════

ACTIONS:

  1. Update constitution to acknowledge pandas pattern?
     → Would make exceptions #1 (data-import) and #1 (legacy-connector)
       no longer necessary

  2. Review exception justifications?
     → Ensure all exceptions have valid, current reasons

  3. No action needed?
     → Current exceptions are appropriate and documented

What would you like to do? [1/2/3]
```

---

## CONSTITUTION UPDATE FLOW

If user chooses to update constitution:

```
📝 UPDATING CONSTITUTION

Current standard (CLAUDE.md §3.2):
  "Use polars for all dataframe operations."

Proposed update:
  "Use polars for dataframe operations. pandas is acceptable
   when interfacing with legacy systems or third-party libraries
   that require it."

This would affect:
  • feat-data-import #1: Would become standard-compliant
  • feat-legacy-connector #1: Would become standard-compliant

Proceed with this update? [Y/n]
```

After update:
```
✅ CONSTITUTION UPDATED

CLAUDE.md §3.2 has been updated.

The following exceptions are now OBSOLETE and can be removed:
  • feat-data-import #1: pandas for legacy API
  • feat-legacy-connector #1: pandas for legacy system

Would you like me to:
  1. Remove these exceptions from the specs
  2. Mark them as [OBSOLETE] but keep for history
  3. Leave them as-is for now
```

---

## NO EXCEPTIONS

If no exceptions exist:

```
╔══════════════════════════════════════════════════════════════════╗
║                    SDD Exceptions Overview                        ║
╚══════════════════════════════════════════════════════════════════╝

✅ NO EXCEPTIONS FOUND

All specifications follow project standards without deviation.

This could mean:
  • Standards are well-suited to project needs
  • Specs are new and haven't encountered edge cases yet
  • Exceptions exist but weren't documented (check specs!)

Run /sdd-status for specification overview.
```

---

## KEY BEHAVIORS

1. **Aggregate across specs** — Show project-wide view
2. **Detect patterns** — Same exception = potential standard update
3. **Track obsolescence** — Constitution changes affect exceptions
4. **Recommend actions** — Don't just report, suggest improvements
5. **Support evolution** — Help constitution grow from real usage
6. **Maintain history** — Exceptions tell the story of the project
