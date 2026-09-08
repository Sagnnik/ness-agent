COMPACTION = """Create a concise continuation checkpoint for the conversation above.

Preserve actionable state and facts. Distinguish commands that completed with a known exit status from partial output and unverified observations. Discard conversational filler and raw tool output unless it contains an important result or error.

Use these exact sections:

**Original Request:** [The user's request]
**Constraints & Acceptance Criteria:** [Strict requirements and definition of done]
**Completed Work:** [Finished work and decisions with reasons]
**Current Operation:** [What is in progress now]
**Key Files:** [Files modified or inspected and their current status]
**Commands & Jobs:** [Commands, process or job IDs, log paths, and current status]
**Verified Results:** [Results backed by a completed command and its exit status]
**Partial Observations:** [Unverified output, incomplete commands, and tentative findings]
**Errors & Rejected Approaches:** [Failures and approaches that must not be repeated]
**Next Step:** [The immediate continuation action]

Do not call tools. Output only the summary."""
