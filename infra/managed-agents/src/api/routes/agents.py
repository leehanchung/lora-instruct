"""Agent CRUD endpoints."""

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db
from src.api.models import Agent, AgentCreate, ListResponse, ModelConfig

router = APIRouter(prefix="/v1/agents", tags=["agents"])

# TODO: Replace with actual ORM models once database schema is defined
_agents_store = {}  # In-memory store for demo


async def _generate_agent_id() -> str:
    """Generate a unique agent ID."""
    return f"agent_{uuid.uuid4().hex[:12]}"


def _serialize_agent(agent_dict: dict) -> Agent:
    """Convert stored agent dict to Agent model."""
    model = agent_dict["model"]
    if isinstance(model, str):
        model = ModelConfig(id=model)
    elif isinstance(model, dict):
        model = ModelConfig(**model)

    return Agent(
        id=agent_dict["id"],
        version=agent_dict.get("version", 1),
        name=agent_dict["name"],
        model=model,
        system=agent_dict["system"],
        tools=agent_dict.get("tools", []),
        custom_tools=agent_dict.get("custom_tools", []),
        mcp_servers=agent_dict.get("mcp_servers", []),
        description=agent_dict.get("description"),
        created_at=agent_dict.get("created_at"),
        updated_at=agent_dict.get("updated_at"),
    )


@router.post("", response_model=Agent, status_code=status.HTTP_201_CREATED)
async def create_agent(
    req: AgentCreate,
    db: AsyncSession = Depends(get_db),
) -> Agent:
    """Create a new Agent.

    Auto-generates ID (agent_xxxx) and sets version to 1.
    """
    from datetime import datetime

    agent_id = await _generate_agent_id()

    # Normalize model
    if isinstance(req.model, str):
        model_obj = ModelConfig(id=req.model)
    else:
        model_obj = req.model

    agent_dict = {
        "id": agent_id,
        "version": 1,
        "name": req.name,
        "model": model_obj.model_dump(),
        "system": req.system,
        "tools": [t.model_dump() for t in req.tools],
        "custom_tools": [t.model_dump() for t in req.custom_tools],
        "mcp_servers": [m.model_dump() for m in req.mcp_servers],
        "description": req.description,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }

    # TODO: Store in database via ORM
    _agents_store[agent_id] = agent_dict

    return _serialize_agent(agent_dict)


@router.get("", response_model=ListResponse)
async def list_agents(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> ListResponse:
    """List agents with pagination."""
    # TODO: Query from database with proper pagination
    agents = [_serialize_agent(a) for a in list(_agents_store.values())[offset : offset + limit]]

    return ListResponse(
        data=agents,
        next_page=None,  # TODO: Calculate based on offset/limit
    )


@router.get("/{agent_id}", response_model=Agent)
async def get_agent(
    agent_id: str,
    db: AsyncSession = Depends(get_db),
) -> Agent:
    """Retrieve an agent by ID."""
    if agent_id not in _agents_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found",
        )

    return _serialize_agent(_agents_store[agent_id])


@router.post("/{agent_id}", response_model=Agent)
async def update_agent(
    agent_id: str,
    req: AgentCreate,
    db: AsyncSession = Depends(get_db),
) -> Agent:
    """Update an agent and bump its version.

    Creates a new version; the original is immutable.
    """
    from datetime import datetime

    if agent_id not in _agents_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found",
        )

    existing = _agents_store[agent_id]
    new_version = existing.get("version", 1) + 1

    # Normalize model
    if isinstance(req.model, str):
        model_obj = ModelConfig(id=req.model)
    else:
        model_obj = req.model

    updated = {
        "id": agent_id,
        "version": new_version,
        "name": req.name,
        "model": model_obj.model_dump(),
        "system": req.system,
        "tools": [t.model_dump() for t in req.tools],
        "custom_tools": [t.model_dump() for t in req.custom_tools],
        "mcp_servers": [m.model_dump() for m in req.mcp_servers],
        "description": req.description,
        "created_at": existing.get("created_at"),
        "updated_at": datetime.utcnow(),
    }

    # TODO: Store versioned agent in database
    _agents_store[agent_id] = updated

    return _serialize_agent(updated)


@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(
    agent_id: str,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete an agent."""
    if agent_id not in _agents_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found",
        )

    # TODO: Soft delete in database (mark archived)
    del _agents_store[agent_id]
