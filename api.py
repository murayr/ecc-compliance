"""HTTP API for the ECC assistant.

    uvicorn api:app --port 8000

POST /ask  {"question": "..."}  ->  answer, cited controls (with their text), token usage
GET  /health
"""
import os
from contextlib import asynccontextmanager
from functools import lru_cache

import anthropic
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import rag


@asynccontextmanager
async def lifespan(app):
    rag.load_index()      # load the index and the embedding model once, at startup,
    rag.embedder()        # so the first request isn't slow
    yield


app = FastAPI(title="ECC assistant", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(","),
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)


@lru_cache(maxsize=1)
def get_client():
    return anthropic.Anthropic()      # reads ANTHROPIC_API_KEY from the environment


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class Control(BaseModel):
    id: str
    domain: str
    subdomain_id: str
    subdomain: str
    text: str
    score: float


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cost_usd: float | None


class AskResponse(BaseModel):
    question: str
    lang: str
    answer: str
    refused: bool
    cited: list[Control]
    model: str
    usage: Usage


@app.get("/health")
def health():
    return {"status": "ok", "model": rag.MODEL}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, client=Depends(get_client)):
    # A plain `def` endpoint: FastAPI runs it in a thread pool, so the blocking SDK call is fine.
    try:
        result = rag.ask(req.question.strip(), client=client)
    except anthropic.RateLimitError:
        raise HTTPException(429, "The model is rate limited. Try again in a minute.")
    except anthropic.APIStatusError as e:
        error = e.body.get("error", {}) if isinstance(e.body, dict) else {}
        raise HTTPException(502, f"Claude API error ({e.status_code}): {error.get('message', e.message)}")
    except anthropic.APIConnectionError:
        raise HTTPException(503, "Could not reach the Claude API.")
    usage = result["usage"]
    print(f"[ask] lang={result['lang']} refused={result['refused']} "
          f"cited={[c['id'] for c in result['cited']]} {rag.format_usage(usage)}", flush=True)
    return result
