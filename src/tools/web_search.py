"""
Web Search Tool
Uses Serper API (if configured) or DuckDuckGo as fallback.
Returns structured chunks from real-time search results.
"""
import os
import json
import time
from typing import List

import requests
from duckduckgo_search import DDGS

from utils.logger import get_logger

logger = get_logger(__name__)
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
SERPER_URL = "https://google.serper.dev/search"
WEB_TIMEOUT = 5.0


def web_search(query: str, max_results: int = 5) -> List[dict]:
    """
    Searches the web for real-time information.
    Tries Serper API first, falls back to DuckDuckGo.
    Returns list of {text, source, score} chunks.
    """
    if SERPER_API_KEY:
        return _serper_search(query, max_results)
    else:
        return _duckduckgo_search(query, max_results)


def _serper_search(query: str, max_results: int) -> List[dict]:
    try:
        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json",
        }
        payload = {"q": query, "num": max_results}
        response = requests.post(SERPER_URL, headers=headers, json=payload, timeout=WEB_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("organic", [])[:max_results]:
            snippet = item.get("snippet", "")
            link = item.get("link", "")
            title = item.get("title", "")
            if snippet:
                results.append({
                    "text": f"{title}\n{snippet}",
                    "source": link,
                    "score": 0.7,
                    "retrieval_method": "web_serper",
                })

        logger.info("serper_search", query=query[:60], results=len(results))
        return results

    except Exception as e:
        logger.warning("serper_search_error", error=str(e))
        return _duckduckgo_search(query, max_results)


def _duckduckgo_search(query: str, max_results: int) -> List[dict]:
    try:
        ddgs = DDGS()
        raw_results = list(ddgs.text(query, max_results=max_results))

        results = []
        for item in raw_results:
            text = f"{item.get('title', '')}\n{item.get('body', '')}"
            href = item.get("href", "")
            if text.strip():
                results.append({
                    "text": text,
                    "source": href,
                    "score": 0.6,
                    "retrieval_method": "web_ddg",
                })

        logger.info("ddg_search", query=query[:60], results=len(results))
        return results

    except Exception as e:
        logger.warning("ddg_search_error", error=str(e))
        return []
