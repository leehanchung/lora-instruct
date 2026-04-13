"""Web fetching and searching tools."""

import asyncio
import logging
import os
import re
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from . import Tool, ToolResult

logger = logging.getLogger(__name__)


class WebFetchTool(Tool):
    """Fetch and parse web content."""

    name = "web_fetch"
    description = "Fetch and parse content from a URL. Converts HTML to plain text."
    input_schema = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to fetch",
            },
            "timeout": {
                "type": "integer",
                "description": "Request timeout in seconds (default 30, max 60)",
            },
        },
        "required": ["url"],
    }

    async def execute(self, input: dict[str, Any]) -> ToolResult:
        """Fetch and parse web content."""
        url = input.get("url", "").strip()
        timeout = input.get("timeout", 30)
        timeout = min(60, max(5, int(timeout)))

        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        logger.info(f"Fetching URL: {url}")

        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                response = await client.get(url)
                response.raise_for_status()

                # Parse HTML to text
                soup = BeautifulSoup(response.text, "html.parser")

                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()

                # Get text
                text = soup.get_text()

                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                text = "\n".join(line for line in lines if line)

                # Limit output size
                if len(text) > 50000:
                    text = text[:50000] + "\n... (truncated)"

                logger.debug(f"Fetched {len(response.text)} bytes from {url}")
                return ToolResult(output=text)

        except httpx.RequestError as e:
            logger.error(f"HTTP error fetching {url}: {e}")
            return ToolResult(output="", error=f"HTTP error: {e}", exit_code=1)
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}", exc_info=True)
            return ToolResult(output="", error=str(e), exit_code=1)


class WebSearchTool(Tool):
    """Search the web."""

    name = "web_search"
    description = "Search the web for information."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query",
            },
            "limit": {
                "type": "integer",
                "description": "Number of results to return (default 5, max 20)",
            },
        },
        "required": ["query"],
    }

    async def execute(self, input: dict[str, Any]) -> ToolResult:
        """Search the web."""
        query = input.get("query", "").strip()
        limit = input.get("limit", 5)
        limit = min(20, max(1, int(limit)))

        if not query:
            return ToolResult(output="", error="Query cannot be empty", exit_code=1)

        logger.info(f"Searching for: {query}")

        # Try SearXNG first (if configured)
        searxng_url = os.getenv("SEARXNG_URL")
        if searxng_url:
            result = await self._search_searxng(query, limit, searxng_url)
            if result.exit_code == 0:
                return result

        # Fallback to DuckDuckGo
        return await self._search_duckduckgo(query, limit)

    async def _search_searxng(
        self, query: str, limit: int, searxng_url: str
    ) -> ToolResult:
        """Search using SearXNG instance.

        Args:
            query: Search query
            limit: Max results
            searxng_url: SearXNG base URL

        Returns:
            ToolResult with search results
        """
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    f"{searxng_url}/search",
                    params={
                        "q": query,
                        "format": "json",
                        "limit": limit,
                    },
                )
                response.raise_for_status()
                data = response.json()

                results = []
                for result in data.get("results", [])[:limit]:
                    results.append(
                        f"- {result.get('title', 'No title')}\n"
                        f"  URL: {result.get('url', 'No URL')}\n"
                        f"  {result.get('content', 'No content')}"
                    )

                if not results:
                    return ToolResult(output="No results found", exit_code=0)

                output = "\n\n".join(results)
                logger.debug(f"SearXNG search returned {len(results)} results")
                return ToolResult(output=output)

        except Exception as e:
            logger.warning(f"SearXNG search failed: {e}")
            return ToolResult(output="", error=str(e), exit_code=1)

    async def _search_duckduckgo(self, query: str, limit: int) -> ToolResult:
        """Search using DuckDuckGo HTML scraping.

        Args:
            query: Search query
            limit: Max results

        Returns:
            ToolResult with search results
        """
        try:
            # Use DuckDuckGo lite (simpler HTML)
            search_url = "https://lite.duckduckgo.com/lite/"

            async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
                response = await client.get(
                    search_url,
                    params={"q": query, "kp": "-1"},  # no safe search
                )
                response.raise_for_status()

                soup = BeautifulSoup(response.text, "html.parser")

                # Extract search results from lite version
                results = []
                result_rows = soup.find_all("tr")

                for row in result_rows:
                    cells = row.find_all("td")
                    if len(cells) >= 2:
                        # First cell usually has link, second has description
                        link_cell = cells[0].find("a")
                        if link_cell:
                            title = link_cell.get_text(strip=True)
                            url = link_cell.get("href", "")

                            # DuckDuckGo lite includes actual URLs
                            if url and (url.startswith("http://") or url.startswith("https://")):
                                description = cells[1].get_text(strip=True) if len(cells) > 1 else ""

                                results.append(
                                    f"- {title}\n"
                                    f"  URL: {url}\n"
                                    f"  {description}"
                                )

                                if len(results) >= limit:
                                    break

                if not results:
                    return ToolResult(output="No results found", exit_code=0)

                output = "\n\n".join(results[:limit])
                logger.debug(f"DuckDuckGo search returned {len(results)} results")
                return ToolResult(output=output)

        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}", exc_info=True)
            return ToolResult(output="", error=str(e), exit_code=1)
