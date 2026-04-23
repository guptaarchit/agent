"""Checkpoint saver for plain Redis (no RediSearch module needed).

`langgraph-checkpoint-redis` requires Redis Stack because it uses redisvl's
search indexes (`FT._LIST` and friends). That's a non-starter on stock Redis.
This module implements the same `BaseCheckpointSaver` interface using only
`HASH`, `ZSET`, and `SET` — commands every Redis 5+ supports.

Storage layout
--------------
    {prefix}:ckpt:{thread}:{ns}:{ckpt_id}    HASH   checkpoint blob + metadata
    {prefix}:ckptidx:{thread}:{ns}           ZSET   member=ckpt_id, score=time
    {prefix}:writes:{thread}:{ns}:{ckpt_id}  HASH   field={task}:{idx}, value=blob
    {prefix}:threads                         SET    all known thread ids

Trade-offs vs the Redis Stack saver
-----------------------------------
- No cross-thread search/filter beyond thread_id
- `alist()` without a thread_id does a full SMEMBERS scan of threads
- No TTL on sessions (add yourself if you want)
"""

from __future__ import annotations

import logging
import pickle
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional, Sequence, Tuple

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from redis.asyncio import Redis


logger = logging.getLogger("agent")


def _decode(v: Any) -> str:
    return v.decode() if isinstance(v, (bytes, bytearray)) else v


class AsyncPlainRedisSaver(BaseCheckpointSaver):
    """Async LangGraph checkpoint saver backed by stock Redis."""

    def __init__(
        self,
        redis: Redis,
        *,
        prefix: str = "lg",
        serde: Optional[SerializerProtocol] = None,
    ) -> None:
        super().__init__(serde=serde or JsonPlusSerializer())
        self.redis = redis
        self.prefix = prefix

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    @asynccontextmanager
    async def from_url(
        cls, url: str, *, prefix: str = "lg"
    ) -> AsyncIterator["AsyncPlainRedisSaver"]:
        redis = Redis.from_url(url)
        try:
            await redis.ping()  # fail fast on unreachable Redis
            logger.info("Connected to Redis at %s", url)
            yield cls(redis, prefix=prefix)
        finally:
            await redis.aclose()

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    def _ckpt_key(self, thread: str, ns: str, ckpt_id: str) -> str:
        return f"{self.prefix}:ckpt:{thread}:{ns}:{ckpt_id}"

    def _ckpt_index_key(self, thread: str, ns: str) -> str:
        return f"{self.prefix}:ckptidx:{thread}:{ns}"

    def _writes_key(self, thread: str, ns: str, ckpt_id: str) -> str:
        return f"{self.prefix}:writes:{thread}:{ns}:{ckpt_id}"

    def _threads_key(self) -> str:
        return f"{self.prefix}:threads"

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    async def aput(
        self,
        config,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ):
        cfg = config["configurable"]
        thread = cfg["thread_id"]
        ns = cfg.get("checkpoint_ns", "")
        ckpt_id = checkpoint["id"]
        parent_id = cfg.get("checkpoint_id") or ""

        ckpt_type, ckpt_blob = self.serde.dumps_typed(checkpoint)
        meta_type, meta_blob = self.serde.dumps_typed(metadata)

        mapping = {
            "type": ckpt_type,
            "checkpoint": ckpt_blob,
            "metadata_type": meta_type,
            "metadata": meta_blob,
            "parent_id": parent_id,
        }

        pipe = self.redis.pipeline()
        pipe.hset(self._ckpt_key(thread, ns, ckpt_id), mapping=mapping)
        pipe.zadd(self._ckpt_index_key(thread, ns), {ckpt_id: time.time()})
        pipe.sadd(self._threads_key(), thread)
        await pipe.execute()

        return {
            "configurable": {
                "thread_id": thread,
                "checkpoint_ns": ns,
                "checkpoint_id": ckpt_id,
            }
        }

    async def aput_writes(
        self,
        config,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        cfg = config["configurable"]
        thread = cfg["thread_id"]
        ns = cfg.get("checkpoint_ns", "")
        ckpt_id = cfg["checkpoint_id"]
        key = self._writes_key(thread, ns, ckpt_id)

        pipe = self.redis.pipeline()
        for idx, (channel, value) in enumerate(writes):
            t, blob = self.serde.dumps_typed(value)
            # Idempotent: same (task, idx) overwrites
            field = f"{task_id}:{idx}"
            payload = pickle.dumps((task_id, task_path, channel, t, blob, idx))
            pipe.hset(key, field, payload)
        await pipe.execute()

    async def aget_tuple(self, config) -> Optional[CheckpointTuple]:
        cfg = config["configurable"]
        thread = cfg["thread_id"]
        ns = cfg.get("checkpoint_ns", "")
        ckpt_id = cfg.get("checkpoint_id")

        if ckpt_id is None:
            ids = await self.redis.zrevrange(self._ckpt_index_key(thread, ns), 0, 0)
            if not ids:
                return None
            ckpt_id = _decode(ids[0])

        raw = await self.redis.hgetall(self._ckpt_key(thread, ns, ckpt_id))
        if not raw:
            return None
        data = {_decode(k): v for k, v in raw.items()}

        checkpoint = self.serde.loads_typed((_decode(data["type"]), data["checkpoint"]))
        metadata = self.serde.loads_typed(
            (_decode(data["metadata_type"]), data["metadata"])
        )
        parent_id = _decode(data.get("parent_id", "")) or None

        pending: list = []
        write_raw = await self.redis.hgetall(self._writes_key(thread, ns, ckpt_id))
        if write_raw:
            rows = []
            for _, payload in write_raw.items():
                rows.append(pickle.loads(payload))
            rows.sort(key=lambda r: (r[0], r[5]))  # (task_id, idx)
            for task_id, _, channel, type_, blob, _ in rows:
                value = self.serde.loads_typed((type_, blob))
                pending.append((task_id, channel, value))

        parent_config = None
        if parent_id:
            parent_config = {
                "configurable": {
                    "thread_id": thread,
                    "checkpoint_ns": ns,
                    "checkpoint_id": parent_id,
                }
            }

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread,
                    "checkpoint_ns": ns,
                    "checkpoint_id": ckpt_id,
                }
            },
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending,
        )

    async def alist(
        self,
        config,
        *,
        filter=None,
        before=None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        # Determine which threads to walk
        cfg = (config or {}).get("configurable", {})
        thread_id = cfg.get("thread_id")
        ns = cfg.get("checkpoint_ns", "")

        if thread_id:
            threads = [thread_id]
        else:
            raw_threads = await self.redis.smembers(self._threads_key())
            threads = [_decode(t) for t in raw_threads]

        before_id = None
        if before:
            before_id = before.get("configurable", {}).get("checkpoint_id")

        yielded = 0
        for t in threads:
            ids = await self.redis.zrevrange(self._ckpt_index_key(t, ns), 0, -1)
            for raw in ids:
                cid = _decode(raw)
                if before_id and cid >= before_id:
                    continue
                tup = await self.aget_tuple(
                    {
                        "configurable": {
                            "thread_id": t,
                            "checkpoint_ns": ns,
                            "checkpoint_id": cid,
                        }
                    }
                )
                if not tup:
                    continue
                if filter:
                    if not all(tup.metadata.get(k) == v for k, v in filter.items()):
                        continue
                yield tup
                yielded += 1
                if limit is not None and yielded >= limit:
                    return

    async def adelete_thread(self, thread_id: str) -> None:
        """Remove every key associated with a thread (all namespaces)."""
        pattern_ckpt = f"{self.prefix}:ckpt:{thread_id}:*"
        pattern_idx = f"{self.prefix}:ckptidx:{thread_id}:*"
        pattern_writes = f"{self.prefix}:writes:{thread_id}:*"
        pipe = self.redis.pipeline()
        for pattern in (pattern_ckpt, pattern_idx, pattern_writes):
            async for key in self.redis.scan_iter(match=pattern):
                pipe.delete(key)
        pipe.srem(self._threads_key(), thread_id)
        await pipe.execute()

    # ------------------------------------------------------------------
    # Sync stubs — this saver is async-only
    # ------------------------------------------------------------------

    def get_tuple(self, config):
        raise NotImplementedError("AsyncPlainRedisSaver is async-only; use aget_tuple")

    def list(self, config, **kwargs):
        raise NotImplementedError("AsyncPlainRedisSaver is async-only; use alist")

    def put(self, *args, **kwargs):
        raise NotImplementedError("AsyncPlainRedisSaver is async-only; use aput")

    def put_writes(self, *args, **kwargs):
        raise NotImplementedError("AsyncPlainRedisSaver is async-only; use aput_writes")
