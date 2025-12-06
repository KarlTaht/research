# Research Agent - Implementation Plan

## Overview

Build a research assistant agent with:
- **Claude Agent SDK** for agent loop (not custom ReAct)
- **`@tool` decorator** for MCP tool definitions
- Full knowledge graph (papers, hypotheses, entities)
- Safety via SDK hooks (`PreToolUse`, `can_use_tool`)

## Architecture Decision: Claude Agent SDK

Using the SDK because:
1. Built-in agent loop (think→act→observe handled by Claude Code)
2. `@tool` decorator teaches MCP patterns without protocol boilerplate
3. `PreToolUse` hooks map directly to our safety requirements
4. Focus effort on the interesting parts: knowledge graph, hypothesis tracking

---

## Phase 1: Foundation

### Step 1: Project Setup

**Files to create:**
- `pyproject.toml` - dependencies and package config
- `src/research_agent/__init__.py`
- `src/research_agent/main.py` - ClaudeSDKClient runner

**Dependencies:**
```
claude-agent-sdk
networkx
sentence-transformers
pdfplumber
httpx
rich
```

### Step 2: Memory Schemas

**File:** `src/research_agent/memory/schemas.py`

```python
@dataclass
class Paper:
    id: str
    title: str
    abstract: str
    key_findings: list[str]
    embedding: np.ndarray | None
    supports: list[str]      # paper IDs
    contradicts: list[str]   # paper IDs

@dataclass
class Hypothesis:
    id: str
    statement: str
    confidence: float        # 0.0 - 1.0
    supporting_evidence: list[str]
    contradicting_evidence: list[str]
    history: list[dict]      # track changes

@dataclass
class Entity:
    id: str
    name: str
    type: str                # method, concept, researcher
    mentioned_in: list[str]
    related_to: list[str]
```

### Step 3: SQLite Storage

**File:** `src/research_agent/memory/store.py`

CRUD operations for papers, hypotheses, entities.

### Step 4: Paper Search Tool

**File:** `src/research_agent/tools/paper_search.py`

```python
from claude_agent_sdk import tool

@tool("search_papers", "Search Semantic Scholar for academic papers",
      {"query": str, "limit": int})
async def search_papers(args):
    results = await semantic_scholar_search(args["query"], args["limit"])
    return {"content": [{"type": "text", "text": json.dumps(results)}]}
```

### Step 5: MCP Server Setup

**File:** `src/research_agent/server.py`

```python
from claude_agent_sdk import create_sdk_mcp_server
from .tools.paper_search import search_papers

research_server = create_sdk_mcp_server(
    name="research",
    version="0.1.0",
    tools=[search_papers]
)
```

### Step 6: Basic Agent Runner

**File:** `src/research_agent/main.py`

```python
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
from .server import research_server

async def run_research(query: str):
    options = ClaudeAgentOptions(
        mcp_servers={"research": research_server},
        allowed_tools=["mcp__research__search_papers"]
    )

    async with ClaudeSDKClient(options) as client:
        await client.query(f"Research this topic: {query}")
        async for message in client.receive_response():
            # Display progress
            pass
```

**Milestone 1:** Agent can search papers via Semantic Scholar

---

## Phase 2: Knowledge Building

### Step 7: Paper Reader Tool

**File:** `src/research_agent/tools/paper_reader.py`

- Download PDF from arXiv URL
- Extract text with pdfplumber
- Return chunked content

### Step 8: Knowledge Graph Tools

**File:** `src/research_agent/tools/knowledge.py`

```python
@tool("add_paper", "Store a paper in the knowledge graph", {...})
@tool("add_hypothesis", "Create a new hypothesis", {...})
@tool("update_confidence", "Update hypothesis confidence", {...})
@tool("find_related", "Find related papers/entities", {...})
@tool("add_entity", "Add entity to knowledge graph", {...})
```

### Step 9: NetworkX Graph

**File:** `src/research_agent/memory/graph.py`

Track relationships:
- Paper → Paper (supports/contradicts)
- Paper → Entity (mentions)
- Hypothesis → Paper (evidence)

### Step 10: Embeddings

**File:** `src/research_agent/memory/embeddings.py`

- sentence-transformers for semantic search
- Store embeddings in SQLite as blob

**Milestone 2:** Agent can read papers, generate hypotheses, build knowledge graph

---

## Phase 3: Safety & Observability

### Step 11: Safety Hooks

**File:** `src/research_agent/safety/hooks.py`

```python
from claude_agent_sdk import HookContext

async def validate_tool_use(input_data: dict, tool_use_id: str, context: HookContext):
    tool_name = input_data.get("tool_name")

    # Block destructive operations
    if "delete" in tool_name or "clear" in tool_name:
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": "Destructive operations blocked"
            }
        }

    # Require confirmation for large confidence changes
    if tool_name == "update_confidence":
        delta = abs(input_data["tool_input"].get("delta", 0))
        if delta >= 0.3:
            # Could trigger human confirmation here
            pass

    return {}
```

### Step 12: Audit Log

**File:** `src/research_agent/safety/audit.py`

- Log all tool calls via `PostToolUse` hook
- Before/after state for knowledge changes
- Periodic SQLite snapshots for rollback

### Step 13: Terminal UI

**File:** `src/research_agent/ui/terminal.py`

Rich-based display:
- Live agent thoughts (from AssistantMessage)
- Tool calls and results (from ToolUseBlock)
- Cost tracking (from ResultMessage.total_cost_usd)

**Milestone 3:** Safe, observable agent with audit trail

---

## Phase 4: Polish

### Step 14: Claude Desktop Integration

Create config for using tools from Claude Desktop:

**File:** `claude_desktop_config.json`
```json
{
  "mcpServers": {
    "research-agent": {
      "command": "python",
      "args": ["-m", "research_agent.server"]
    }
  }
}
```

### Step 15: End-to-End Testing

Test with real research question:
"What are the main challenges in long-horizon agentic systems?"

Verify:
- Papers are found and stored
- Hypotheses are generated with evidence
- Confidence updates work correctly
- Audit log captures all actions

### Step 16: Documentation

- README with architecture diagram
- Usage examples (CLI + Claude Desktop)
- Design decisions document (safety architecture)

**Milestone 4:** Portfolio-ready project

---

## File Summary

```
src/research_agent/
├── __init__.py
├── main.py                    # ClaudeSDKClient runner
├── server.py                  # create_sdk_mcp_server()
├── tools/
│   ├── __init__.py
│   ├── paper_search.py        # @tool: Semantic Scholar
│   ├── paper_reader.py        # @tool: PDF extraction
│   └── knowledge.py           # @tool: graph operations
├── memory/
│   ├── __init__.py
│   ├── schemas.py             # Paper, Hypothesis, Entity
│   ├── store.py               # SQLite CRUD
│   ├── graph.py               # NetworkX relationships
│   └── embeddings.py          # sentence-transformers
├── safety/
│   ├── __init__.py
│   ├── hooks.py               # PreToolUse/PostToolUse
│   └── audit.py               # Action logging
└── ui/
    ├── __init__.py
    └── terminal.py            # Rich display
```

---

## Open Decisions (Defer)

- Multi-session task persistence (how does agent resume research?)
- Web UI (FastAPI + React, stretch goal)
- Contradiction detection approach (LLM-based, flag for human review)
- Multi-model routing (SDK handles model, but could customize prompts per task type)
