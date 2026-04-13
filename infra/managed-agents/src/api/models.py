"""Pydantic models for the Agent API, mirroring Claude Managed Agents."""

from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


# ============================================================================
# Enums
# ============================================================================


class LLMBackendType(str, Enum):
    """LLM backend: either Claude API or self-hosted vLLM."""
    ANTHROPIC = "anthropic"
    VLLM = "vllm"


class SessionStatus(str, Enum):
    """Session lifecycle status."""
    RESCHEDULING = "rescheduling"
    RUNNING = "running"
    IDLE = "idle"
    TERMINATED = "terminated"


class EnvironmentType(str, Enum):
    """Environment container type."""
    CLOUD = "cloud"


class NetworkingType(str, Enum):
    """Networking policy for environment."""
    UNRESTRICTED = "unrestricted"
    LIMITED = "limited"


class EventType(str, Enum):
    """Event types in session event stream."""
    USER_MESSAGE = "user.message"
    USER_INTERRUPT = "user.interrupt"
    USER_CUSTOM_TOOL_RESULT = "user.custom_tool_result"
    AGENT_MESSAGE = "agent.message"
    AGENT_TOOL_USE = "agent.tool_use"
    AGENT_TOOL_RESULT = "agent.tool_result"
    AGENT_CUSTOM_TOOL_USE = "agent.custom_tool_use"
    SESSION_STATUS_IDLE = "session.status_idle"


class ToolType(str, Enum):
    """Tool definition type."""
    BUILT_IN = "builtin"
    CUSTOM = "custom"


# ============================================================================
# Model Configuration
# ============================================================================


class ModelConfig(BaseModel):
    """LLM model configuration."""
    id: str = Field(..., description="Model identifier (e.g., claude-3-5-sonnet-20241022)")
    backend: LLMBackendType = Field(default=LLMBackendType.ANTHROPIC)
    speed: Optional[str] = Field(default=None, description="Speed tier for API-based models")

    class Config:
        use_enum_values = False


# ============================================================================
# Tool Definitions
# ============================================================================


class ToolDefinition(BaseModel):
    """Tool definition with optional configuration."""
    name: str = Field(..., description="Tool name")
    description: Optional[str] = Field(default=None)
    input_schema: Optional[Dict[str, Any]] = Field(default=None)


class AgentToolset(BaseModel):
    """Built-in agent toolset with optional configuration."""
    name: str = Field(..., description="Toolset name (e.g., code_execution)")
    config: Optional[Dict[str, Any]] = Field(default=None)


# ============================================================================
# MCP Server
# ============================================================================


class MCPServer(BaseModel):
    """Model Context Protocol server configuration."""
    name: str = Field(..., description="MCP server name")
    url: Optional[str] = Field(default=None, description="Server URL if remote")
    command: Optional[str] = Field(default=None, description="Local server command")
    args: Optional[List[str]] = Field(default=None)
    env: Optional[Dict[str, str]] = Field(default=None)


# ============================================================================
# Agent
# ============================================================================


class AgentCreate(BaseModel):
    """Request to create an Agent."""
    name: str = Field(..., description="Human-readable agent name")
    model: str | ModelConfig = Field(
        ..., description="Model ID string or ModelConfig object"
    )
    system: str = Field(..., description="System prompt for agent")
    tools: List[AgentToolset] = Field(default_factory=list, description="Built-in toolsets")
    custom_tools: List[ToolDefinition] = Field(default_factory=list, description="Custom tools")
    mcp_servers: List[MCPServer] = Field(default_factory=list, description="MCP servers")
    description: Optional[str] = Field(default=None)


class Agent(BaseModel):
    """Agent response model."""
    id: str = Field(..., description="Agent ID (agent_xxxx)")
    version: int = Field(default=1, description="Version number")
    name: str
    model: ModelConfig
    system: str
    tools: List[AgentToolset]
    custom_tools: List[ToolDefinition]
    mcp_servers: List[MCPServer]
    description: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ============================================================================
# Environment
# ============================================================================


class NetworkingConfig(BaseModel):
    """Networking configuration for environment."""
    type: NetworkingType = Field(default=NetworkingType.UNRESTRICTED)
    allowed_hosts: Optional[List[str]] = Field(
        default=None, description="Allowed hosts if type=limited"
    )


class EnvironmentConfig(BaseModel):
    """Environment container configuration."""
    type: EnvironmentType = Field(default=EnvironmentType.CLOUD)
    packages: Dict[str, List[str]] = Field(
        default_factory=dict, description="Packages by manager (pip, apt, etc.)"
    )
    networking: NetworkingConfig = Field(default_factory=NetworkingConfig)


class EnvironmentCreate(BaseModel):
    """Request to create an Environment."""
    name: str = Field(..., description="Human-readable environment name")
    config: EnvironmentConfig = Field(default_factory=EnvironmentConfig)


class Environment(BaseModel):
    """Environment response model."""
    id: str = Field(..., description="Environment ID")
    name: str
    config: EnvironmentConfig
    created_at: datetime
    updated_at: datetime
    archived_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ============================================================================
# Session
# ============================================================================


class AgentRef(BaseModel):
    """Reference to an Agent (can be string ID or object)."""
    id: str = Field(..., description="Agent ID")


class SessionResources(BaseModel):
    """Resources available to session."""
    github_repository: Optional[str] = None
    file: Optional[str] = None


class SessionCreate(BaseModel):
    """Request to create a Session."""
    agent: str | AgentRef = Field(..., description="Agent ID or AgentRef")
    environment_id: str = Field(..., description="Environment ID")
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default=None)
    resources: Optional[SessionResources] = None


class SessionStats(BaseModel):
    """Session statistics."""
    total_events: int = 0
    user_messages: int = 0
    agent_messages: int = 0
    tool_uses: int = 0


class SessionUsage(BaseModel):
    """Session resource usage."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class Session(BaseModel):
    """Session response model."""
    id: str = Field(..., description="Session ID")
    type: str = Field(default="agent", description="Session type")
    status: SessionStatus
    agent: Agent = Field(..., description="Resolved agent object")
    environment_id: str
    title: Optional[str]
    metadata: Optional[Dict[str, Any]]
    resources: Optional[SessionResources]
    created_at: datetime
    updated_at: datetime
    archived_at: Optional[datetime] = None
    stats: SessionStats = Field(default_factory=SessionStats)
    usage: SessionUsage = Field(default_factory=SessionUsage)

    class Config:
        from_attributes = True


# ============================================================================
# Events
# ============================================================================


class UserMessageEvent(BaseModel):
    """User sends a text message."""
    type: str = Field(default=EventType.USER_MESSAGE)
    content: str = Field(..., description="Message text")


class UserInterruptEvent(BaseModel):
    """User interrupts the agent."""
    type: str = Field(default=EventType.USER_INTERRUPT)


class UserCustomToolResultEvent(BaseModel):
    """User provides result for a custom tool use."""
    type: str = Field(default=EventType.USER_CUSTOM_TOOL_RESULT)
    tool_use_id: str = Field(..., description="ID of agent's custom tool use")
    content: Any = Field(..., description="Tool result content")


class AgentMessageEvent(BaseModel):
    """Agent sends a text message."""
    type: str = Field(default=EventType.AGENT_MESSAGE)
    content: str = Field(..., description="Message text")
    id: str = Field(default_factory=lambda: f"msg_{id(object())}", description="Message ID")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AgentToolUseEvent(BaseModel):
    """Agent uses a built-in or MCP tool."""
    type: str = Field(default=EventType.AGENT_TOOL_USE)
    id: str = Field(..., description="Tool use ID")
    tool_name: str = Field(..., description="Tool name")
    tool_input: Dict[str, Any] = Field(..., description="Tool input parameters")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AgentToolResultEvent(BaseModel):
    """Result of a tool use."""
    type: str = Field(default=EventType.AGENT_TOOL_RESULT)
    tool_use_id: str = Field(..., description="ID of corresponding tool use")
    content: Any = Field(..., description="Tool result")
    is_error: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AgentCustomToolUseEvent(BaseModel):
    """Agent uses a custom tool requiring user intervention."""
    type: str = Field(default=EventType.AGENT_CUSTOM_TOOL_USE)
    id: str = Field(..., description="Tool use ID")
    tool_name: str = Field(..., description="Custom tool name")
    tool_input: Dict[str, Any] = Field(..., description="Tool input parameters")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SessionStatusIdleEvent(BaseModel):
    """Session transitioned to idle."""
    type: str = Field(default=EventType.SESSION_STATUS_IDLE)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class EventSend(BaseModel):
    """Request to send events to a session."""
    events: List[
        UserMessageEvent
        | UserInterruptEvent
        | UserCustomToolResultEvent
    ] = Field(...)


# ============================================================================
# Pagination
# ============================================================================


class ListResponse(BaseModel):
    """Generic paginated list response."""
    data: List[Any] = Field(...)
    next_page: Optional[str] = Field(
        default=None, description="Cursor for next page, if available"
    )


# ============================================================================
# Error Models
# ============================================================================


class ErrorDetail(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    code: Optional[str] = Field(default=None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(default=None)
