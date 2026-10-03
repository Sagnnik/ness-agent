from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from ness_agent.context.budget import CompactionStatus

class AgentState(TypedDict, total=False):
    """Checkpointed graph state for one thread."""

    # ----- 1. Conversation and Model-facing context -----
    
    # clean full conversation history always
    # includes human, ai, tool messages
    # exlcudes L3 overlay and system message
    # used for persistance, CLI events, compaction input, tool loops etc.
    messages: Annotated[list[BaseMessage], add_messages]  

    # snapshot of the model-facing conversation except system message
    # normal turn -> last invoked payload (may contain L3 overlay)
    # after compaction -> [compacted-history] + active turn
    model_context_messages: list[BaseMessage]

    # how many raw messages does the compacted prefix already count
    # set to len(messages) at compaction time after each agent invoke 
    model_context_source_count: int

    # records the exact system message from the last successful model call
    model_system_message: BaseMessage

    # input token count from the last agent API call (usage_metadata.input_tokens, or estimate fallback)
    last_input_tokens: int


    # ----- 2. Compaction -----
    
    # Result of the latest context_gate pass (even if no compaction happened)
    # Always has: compacted, token_count, ratio, context_limit, overlay_note
    # On success: trigger = "automatic" | "manual" | "safety", plus after_tokens, etc.
    # On skip/fail: skip_reason = "retry_suppressed" | "no_completed_history" | "disabled" | "failed"
    # Lifetime: set in context_gate; cleared at start of each agent_node invoke (compaction_status: {})
    # Consumer: agent_node reads compacted to decide full vs delta L3 overlay injection,
    # and overlay_note for the L3 compaction section.
    compaction_status: CompactionStatus

    # id of the active user message when compaction LLM last failed.
    # suppress re-calling compaction on every tool loop for the same turn; gets cleared after successful compaction
    compaction_failed_turn_id: str
    
    # flag to force compaction even if not needed
    force_compact: bool

    # ----- 3. Mode & skills -----
    
    # "act" or "plan" for this graph run
    # set each turn in session _build_run_payload (mode: self.mode)
    mode: str
    
    # one-shot signal for first act turn after plan→act.
    mode_switch: str
    
    # Skill names requested for this turn by the host application.
    requested_skills: list[str]

    # Complete effective skill catalog to inject once through L3.
    skill_catalog: str

    # ----- 4. Tools & approval -----

    # dict[tool_call_id -> denial reason string] - why was tool call denied?
    # Set by: approval_gate after user/handler denies tools and used by tools_node 
    # reset at each agent node call after tools_node completes
    approval_declined: dict[str, str]
    # Canonical filesystem grants valid only for their approved tool call.
    approval_file_access: dict[str, list[dict]]
    
    # Task list for the thread.
    # Shape: [{id, content, status}] — status: pending | in_progress | completed
    # Updated by: todo tool (via in-memory store synced back in tools_node)
    todos: list[dict]
    
    # Message index — how much of messages was already covered by the last background reflection run; default 0
    # Updated by: agent_node after scheduling reflection (consume_reflection_message_index())
    # Used by: _schedule_reflection_if_due() — only reflects on new messages since this index.
    last_reflection_index: int
