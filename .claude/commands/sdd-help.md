# /sdd-help — SDD Quick Reference

Display the SDD (Specification-Driven Development) workflow overview and available commands.




---

## DISPLAY THIS INFORMATION

When this command is invoked, display the following:

```
╔══════════════════════════════════════════════════════════════════╗
║           SDD - Specification-Driven Development                  ║
╚══════════════════════════════════════════════════════════════════╝

PHILOSOPHY:
  "The specification is the single source of truth."

  When disagreements arise → reference the spec
  When scope creeps → update the spec first
  When bugs appear → trace back to the spec

────────────────────────────────────────────────────────────────────

WORKFLOW:

  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ SPECIFY  │───▶│   PLAN   │───▶│  TASKS   │───▶│IMPLEMENT │
  │          │    │          │    │          │    │          │
  │ What &   │    │ How &    │    │ Ordered  │    │ TDD      │
  │ Why      │    │ Where    │    │ Steps    │    │ Execution│
  └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘
       │               │               │               │
       ▼               ▼               ▼               ▼
  [Human Review] [Human Review] [Human Review]  [Tests Pass]

Each phase requires approval before proceeding to the next.

────────────────────────────────────────────────────────────────────

COMMANDS:

  /sdd-help        You are here - workflow overview
  /sdd-spec        Create or continue a feature specification
  /sdd-plan        Create technical implementation plan
  /sdd-tasks       Break plan into atomic tasks
  /sdd-implement   Execute tasks with TDD
  /sdd-status      Show progress across all specifications
  /sdd-resume      Continue incomplete work
  /sdd-exceptions  List all exceptions with pattern detection

────────────────────────────────────────────────────────────────────

DIRECTORY STRUCTURE:

  .specs/
  ├── templates/           # Templates for new specs
  │   ├── spec-template.md
  │   ├── plan-template.md
  │   └── tasks-template.md
  └── feat-{name}/         # Per-feature directory
      ├── spec.md          # What & Why
      ├── plan.md          # How & Where
      ├── tasks.md         # Ordered steps
      ├── .session.md      # Session context (for resume)
      └── refs/            # Reference materials

────────────────────────────────────────────────────────────────────

QUICK START:

  1. Run /sdd-spec to begin a new specification
  2. Answer the structured interview questions
  3. Review the generated spec.md
  4. Mark as Approved when ready
  5. Run /sdd-plan to design implementation
  6. Run /sdd-tasks to break into steps
  7. Run /sdd-implement to execute with TDD

────────────────────────────────────────────────────────────────────

SPECIFICATION STATUS LEGEND:

  Draft        → Initial creation, not reviewed
  In Progress  → Interview ongoing or refinement needed
  Approved     → Ready for planning/implementation
  Completed    → Feature implemented and verified

────────────────────────────────────────────────────────────────────

CONFLICT RESOLUTION:

  When your request conflicts with project standards (CLAUDE.md):

  1. CONFORM   → Follow the project standard
  2. EXCEPTION → Document deviation with justification
  3. INVESTIGATE → Analyze before deciding

  Philosophy: "Don't block. Track. Evolve."
  Exceptions are tracked, patterns drive constitution updates.

────────────────────────────────────────────────────────────────────

For detailed instructions, see CLAUDE.md (project constitution)
```

---

## ALSO SHOW CURRENT STATUS

After displaying the help, scan `.specs/` and show current status:

```
CURRENT STATUS:
```

If `.specs/` exists and has feature directories:
```
  Found {N} specifications:

  | Feature           | Phase      | Progress | Last Updated |
  |-------------------|------------|----------|--------------|
  | feat-user-auth    | Tasks      | 70%      | 2 hours ago  |
  | feat-dark-mode    | Spec       | 30%      | 3 days ago   |

  Run /sdd-status for full details
```

If no specs exist:
```
  No specifications found.
  Run /sdd-spec to create your first specification.
```

If `.specs/` doesn't exist:
```
  SDD not initialized in this project.
  Run 'sdd init' in your terminal to set up SDD.
```
