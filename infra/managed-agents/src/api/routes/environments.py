"""Environment CRUD endpoints."""

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db
from src.api.models import Environment, EnvironmentCreate, ListResponse

router = APIRouter(prefix="/v1/environments", tags=["environments"])

# TODO: Replace with actual ORM models once database schema is defined
_environments_store = {}  # In-memory store for demo


async def _generate_environment_id() -> str:
    """Generate a unique environment ID."""
    return f"env_{uuid.uuid4().hex[:12]}"


def _serialize_environment(env_dict: dict) -> Environment:
    """Convert stored environment dict to Environment model."""
    return Environment(
        id=env_dict["id"],
        name=env_dict["name"],
        config=env_dict["config"],
        created_at=env_dict.get("created_at"),
        updated_at=env_dict.get("updated_at"),
        archived_at=env_dict.get("archived_at"),
    )


@router.post("", response_model=Environment, status_code=status.HTTP_201_CREATED)
async def create_environment(
    req: EnvironmentCreate,
    db: AsyncSession = Depends(get_db),
) -> Environment:
    """Create a new Environment."""
    env_id = await _generate_environment_id()

    env_dict = {
        "id": env_id,
        "name": req.name,
        "config": req.config.model_dump(),
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "archived_at": None,
    }

    # TODO: Store in database via ORM
    _environments_store[env_id] = env_dict

    return _serialize_environment(env_dict)


@router.get("", response_model=ListResponse)
async def list_environments(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    include_archived: bool = Query(False),
    db: AsyncSession = Depends(get_db),
) -> ListResponse:
    """List environments with pagination."""
    envs = list(_environments_store.values())

    # Filter archived if requested
    if not include_archived:
        envs = [e for e in envs if e.get("archived_at") is None]

    envs = envs[offset : offset + limit]
    environments = [_serialize_environment(e) for e in envs]

    return ListResponse(
        data=environments,
        next_page=None,  # TODO: Calculate based on offset/limit
    )


@router.get("/{environment_id}", response_model=Environment)
async def get_environment(
    environment_id: str,
    db: AsyncSession = Depends(get_db),
) -> Environment:
    """Retrieve an environment by ID."""
    if environment_id not in _environments_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Environment {environment_id} not found",
        )

    return _serialize_environment(_environments_store[environment_id])


@router.post("/{environment_id}", response_model=Environment)
async def update_environment(
    environment_id: str,
    req: EnvironmentCreate,
    db: AsyncSession = Depends(get_db),
) -> Environment:
    """Update an environment."""
    if environment_id not in _environments_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Environment {environment_id} not found",
        )

    existing = _environments_store[environment_id]
    existing.update(
        {
            "name": req.name,
            "config": req.config.model_dump(),
            "updated_at": datetime.utcnow(),
        }
    )

    # TODO: Update in database
    return _serialize_environment(existing)


@router.post("/{environment_id}/archive", response_model=Environment)
async def archive_environment(
    environment_id: str,
    db: AsyncSession = Depends(get_db),
) -> Environment:
    """Archive an environment (soft delete)."""
    if environment_id not in _environments_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Environment {environment_id} not found",
        )

    existing = _environments_store[environment_id]
    existing["archived_at"] = datetime.utcnow()

    # TODO: Update in database
    return _serialize_environment(existing)


@router.delete("/{environment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_environment(
    environment_id: str,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Hard delete an environment."""
    if environment_id not in _environments_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Environment {environment_id} not found",
        )

    # TODO: Hard delete from database
    del _environments_store[environment_id]
