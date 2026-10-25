"""API or local Hugging Face answers, and local Ollama embeddings."""

import json
import re

from langchain_anthropic import ChatAnthropic
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langsmith import tracing_context
from ollama import Client

from document_rag.config import (
    CHAT_MODEL,
    EMBED_INPUT_BYTES,
    EMBED_MODEL,
    HF_CHAT_MODEL,
    OLLAMA_HOST,
    RagError,
    require_env,
)


class RetrievalEmbeddings(Embeddings):
    """Apply EmbeddingGemma's query prompt and reject over-long inputs."""

    def __init__(self, underlying):
        self.underlying = underlying

    def embed_documents(self, texts):
        if any(len(text.encode("utf-8")) > EMBED_INPUT_BYTES for text in texts):
            raise RagError(
                "Embedding input is too long. Re-ingest with the configured text splitter."
            )
        return self.underlying.embed_documents(texts)

    def embed_query(self, text):
        query = "task: search result | query: " + text
        if len(query.encode("utf-8")) > EMBED_INPUT_BYTES:
            raise RagError(
                "Please shorten the question to fit the embedding model's context."
            )
        return self.underlying.embed_query(query)


def parse_generated_json(text):
    """Read the JSON object from a local model reply, including fenced output."""
    raw = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    if fenced:
        raw = fenced.group(1).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object")
    value = json.loads(raw[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("JSON value is not an object")
    return value


class ApiModels:
    """Chat through Anthropic or a local Hugging Face model. Embeddings stay on Ollama."""

    def __init__(self, chat_model=None, provider="anthropic"):
        if provider not in {"anthropic", "huggingface"}:
            raise RagError("Generation provider must be 'anthropic' or 'huggingface'.")
        self.provider = provider
        if chat_model:
            self.chat_model = chat_model
        elif provider == "huggingface":
            self.chat_model = HF_CHAT_MODEL
        else:
            self.chat_model = CHAT_MODEL
        self.client = Client(host=OLLAMA_HOST, timeout=600, trust_env=False)
        self._digests = {}
        self._pipeline = None

    def digest(self, model=EMBED_MODEL):
        if model not in self._digests:
            try:
                available = {
                    item.model: item.digest for item in self.client.list().models
                }
            except Exception as exc:
                raise RagError(
                    "Cannot reach local Ollama. Start it, then run: ollama pull "
                    + EMBED_MODEL
                ) from exc
            if model not in available:
                raise RagError(
                    f"Required local embedding model is missing. Run: ollama pull {model}"
                )
            self._digests[model] = available[model]
        return self._digests[model]

    def embeddings(self):
        self.digest(EMBED_MODEL)
        return RetrievalEmbeddings(
            OllamaEmbeddings(
                model=EMBED_MODEL,
                base_url=OLLAMA_HOST,
                keep_alive=1800,
                client_kwargs={"timeout": 600, "trust_env": False},
            )
        )

    def chat(self, system, question, schema=None, max_tokens=900):
        if self.provider == "huggingface":
            return self._chat_local(system, question, schema, max_tokens)
        return self._chat_anthropic(system, question, schema, max_tokens)

    def _chat_anthropic(self, system, question, schema, max_tokens):
        require_env("ANTHROPIC_API_KEY")
        prompt = ChatPromptTemplate.from_messages(
            [("system", "{system}"), ("human", "{question}")]
        )
        llm = ChatAnthropic(
            model=self.chat_model,
            temperature=0,
            max_tokens=max_tokens,
        )
        runnable = llm.with_structured_output(schema) if schema else llm
        try:
            with tracing_context(enabled=False):
                result = (prompt | runnable).invoke(
                    {"system": system, "question": question}
                )
        except Exception as exc:
            raise RagError(f"Anthropic request failed: {exc}") from exc
        if schema:
            if hasattr(result, "model_dump"):
                result = result.model_dump()
            if not isinstance(result, dict):
                raise RagError(
                    "The model returned invalid structured output. Try rephrasing the question."
                )
            missing = [key for key in (schema.get("required") or []) if key not in result]
            if missing:
                raise RagError(
                    "The model stopped before finishing its answer. Try the question again."
                )
            return result
        return result.content
    def _chat_local(self, system, question, schema, max_tokens):
        if schema is not None:
            system = (
                system
                + "\n\nReturn one JSON object and no other text. "
                + "The object must match this JSON schema:\n"
                + json.dumps(schema)
            )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]
        text = self._complete(messages, max_tokens)
        if schema is None:
            return text.strip()
        try:
            return parse_generated_json(text)
        except (ValueError, json.JSONDecodeError) as exc:
            raise RagError(
                "The local model did not return valid JSON. "
                "Try the Anthropic API, or set HF_CHAT_MODEL to a larger instruct model."
            ) from exc

    def _complete(self, messages, max_tokens):
        pipe = self._local_pipeline()
        tokenizer = pipe.tokenizer
        if getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = "\n\n".join(message["content"] for message in messages)
        try:
            output = pipe(
                prompt,
                max_new_tokens=max_tokens,
                do_sample=False,
                return_full_text=False,
            )
        except Exception as exc:
            raise RagError(f"Local model generation failed: {exc}") from exc
        text = output[0]["generated_text"]
        if isinstance(text, list):
            last = text[-1] if text else ""
            text = last.get("content", "") if isinstance(last, dict) else str(last)
        if not isinstance(text, str) or not text.strip():
            raise RagError("The local model returned an empty response.")
        return text

    def _local_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:
            raise RagError(
                "Local Hugging Face generation is not installed. "
                "From the document-rag folder, run: uv sync --extra local"
            ) from exc
        if torch.cuda.is_available():
            device, dtype = "cuda", torch.float16
        elif torch.backends.mps.is_available():
            device, dtype = "mps", torch.float16
        else:
            device, dtype = "cpu", torch.float32
        try:
            try:
                pipe = pipeline(
                    "text-generation",
                    model=self.chat_model,
                    dtype=dtype,
                    device=device,
                )
            except TypeError:
                pipe = pipeline(
                    "text-generation",
                    model=self.chat_model,
                    torch_dtype=dtype,
                    device=device,
                )
        except Exception as exc:
            raise RagError(
                f"Could not load local model {self.chat_model}: {exc}"
            ) from exc
        tokenizer = pipe.tokenizer
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        self._pipeline = pipe
        return pipe

