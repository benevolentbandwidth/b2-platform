from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
from time import perf_counter
from typing import Optional

from .embeddings import GoogleEmbedder
from .orchestrator import agent as agent_module
from .store import AgentIndexStore, StoredAgentRecord

logger = logging.getLogger(__name__)


class BelowThresholdError(ValueError):
	"""Raised when no route meets the configured minimum score."""

	def __init__(self, score: float, threshold: float):
		self.score = score
		self.threshold = threshold
		super().__init__(f"No agent met routing threshold ({score:.3f} < {threshold:.3f})")


@dataclass
class _CachedAgent:
	agent: agent_module.Agent
	yaml_path: str
	mtime: float


class AgentRouter:
	"""Index agent YAML files and route queries to the best matching agent."""

	def __init__(
		self,
		agents_dir: Optional[str] = None,
		db_path: Optional[str] = None,
		embedder: Optional[GoogleEmbedder] = None,
		min_score: float = 0.35,
	):
		start = perf_counter()
		self.agents_dir = self._agents_dir(agents_dir)
		self.store = AgentIndexStore(db_path=db_path)
		self.embedder = embedder or GoogleEmbedder()
		self.min_score = min_score
		self._agent_cache: dict[str, _CachedAgent] = {}
		self.refresh_index()
		logger.info("router.init done elapsed=%.3fs", perf_counter() - start)

	def _agents_dir(self, agents_dir: Optional[str] = None) -> Path:
		if agents_dir is None:
			project_root = Path(__file__).parent.parent
			return project_root / "agents"
		return Path(agents_dir)

	@staticmethod
	def _hash_content(definition: agent_module.AgentDefinition) -> str:
		content = f"{definition.description}\n---\n{definition.system_prompt}".encode("utf-8")
		return hashlib.sha256(content).hexdigest()

	@staticmethod
	def _load_agent_definition(yaml_path: Path) -> agent_module.AgentDefinition:
		return agent_module._load_agent_definition_from_file(yaml_path)

	@staticmethod
	def _embed_text(definition: agent_module.AgentDefinition) -> str:
		return f"{definition.name}\n{definition.description}\n{definition.system_prompt}"

	def _load_agent_for_record(self, record: StoredAgentRecord) -> agent_module.Agent:
		yaml_path = Path(record.yaml_path)
		mtime = yaml_path.stat().st_mtime
		cached = self._agent_cache.get(record.name)
		if cached and cached.yaml_path == record.yaml_path and cached.mtime == mtime:
			logger.info("router.cache hit name=%s", record.name)
			return cached.agent

		logger.info("router.cache miss name=%s", record.name)
		definition = self._load_agent_definition(yaml_path)
		agent = agent_module.Agent(definition)
		self._agent_cache[record.name] = _CachedAgent(
			agent=agent,
			yaml_path=record.yaml_path,
			mtime=mtime,
		)
		return agent

	def refresh_index(self) -> None:
		start = perf_counter()
		if not self.agents_dir.exists():
			logger.warning(
				"router.refresh_index agents directory not found at %s; keeping existing store rows",
				self.agents_dir,
			)
			return

		existing: dict[str, StoredAgentRecord] = {r.name: r for r in self.store.all_agents()}
		indexed_names: list[str] = []
		to_embed_rows: list[tuple[agent_module.AgentDefinition, Path, str]] = []
		unchanged = 0
		yaml_paths = list(self.agents_dir.glob("*.yaml"))
		if not yaml_paths:
			logger.warning(
				"router.refresh_index found zero YAML files in %s; keeping existing store rows",
				self.agents_dir,
			)
			return

		for yaml_path in yaml_paths:
			definition = self._load_agent_definition(yaml_path)
			indexed_names.append(definition.name)

			content_hash = self._hash_content(definition)

			existing_row = existing.get(definition.name)
			if (
				existing_row
				and existing_row.content_hash == content_hash
				and existing_row.embedding_model == self.embedder.model
				and existing_row.embedding_dim == self.embedder.output_dimensionality
			):
				unchanged += 1
				continue

			to_embed_rows.append((definition, yaml_path, content_hash))

		embed_start = perf_counter()
		embeddings = self.embedder.embed_documents([self._embed_text(row[0]) for row in to_embed_rows])
		logger.info(
			"router.refresh_index embeddings elapsed=%.3fs rows=%d",
			perf_counter() - embed_start,
			len(to_embed_rows),
		)

		for i, (definition, yaml_path, content_hash) in enumerate(to_embed_rows):
			self.store.upsert_agent(
				name=definition.name,
				description=definition.description,
				yaml_path=str(yaml_path),
				embedding=embeddings[i],
				embedding_model=self.embedder.model,
				content_hash=content_hash,
			)

		self.store.delete_missing(indexed_names)
		for cached_name in list(self._agent_cache):
			if cached_name not in indexed_names:
				logger.info("router.cache evict missing name=%s", cached_name)
				self._agent_cache.pop(cached_name, None)
		logger.info(
			"router.refresh_index done elapsed=%.3fs new=%d unchanged=%d",
			perf_counter() - start,
			len(to_embed_rows),
			unchanged,
		)

	def route_with_metadata(
		self,
		query: str,
		current_agent: Optional[str] = None,
	) -> tuple[agent_module.Agent, dict[str, str | float | bool]]:
		"""Return the routed agent plus the match metadata.

		A conversation whose current agent is sticky stays with it. Each message
		is otherwise routed on its own, and a claimant's short follow-ups ("he was
		my dad", a date and place) scored closer to the general agent than to the
		intake agent, moving the claim to an agent that cannot handle it.
		"""
		if current_agent:
			record = next((r for r in self.store.all_agents() if r.name == current_agent), None)
			if record is not None:
				agent = self._load_agent_for_record(record)
				if agent.definition.sticky:
					logger.info("router.route sticky name=%s", record.name)
					return agent, {
						"name": record.name,
						"description": record.description,
						"yaml_path": record.yaml_path,
						"sticky": True,
					}

		start = perf_counter()
		embed_start = perf_counter()
		query_embedding = self.embedder.embed_query(query)
		logger.info("router.route query embedding elapsed=%.3fs", perf_counter() - embed_start)
		match_start = perf_counter()
		matches = self.store.match(query_embedding, limit=1)
		logger.info("router.route store match elapsed=%.3fs", perf_counter() - match_start)
		if not matches:
			raise ValueError("No agents are indexed")

		record, score = matches[0]
		if score < self.min_score:
			raise BelowThresholdError(score, self.min_score)

		load_start = perf_counter()
		agent = self._load_agent_for_record(record)
		logger.info("router.route agent load elapsed=%.3fs name=%s", perf_counter() - load_start, record.name)
		logger.info("router.route done elapsed=%.3fs score=%.3f", perf_counter() - start, score)

		return agent, {
			"name": record.name,
			"description": record.description,
			"yaml_path": record.yaml_path,
			"score": score,
		}
