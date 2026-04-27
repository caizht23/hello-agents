from __future__ import annotations

import importlib
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from dotenv import load_dotenv

from backend.agents.base import BaseAgent
from backend.config import ENV_FILE, ROOT_DIR, settings
from backend.events import event_logger
from backend.maintenance import cleanup_deep_research_artifacts
from backend.memory.base import memory_store
from backend.models import AgentRequest, AgentResponse


class DeepResearchAdapter(BaseAgent):
    """Expose chapter14 DeepResearchAgent as one platform-level agent."""

    def run(self, request: AgentRequest) -> AgentResponse:
        event_logger.emit("agent_started", agent_id=self.agent_id, task_id=request.task_id)
        try:
            output, artifacts = self._run_with_artifacts(request)
        except Exception as exc:
            output = f"deep_research 运行失败：{type(exc).__name__}: {exc}"
            artifacts = {"error": str(exc), "error_type": type(exc).__name__}

        memory_store.add(self.agent_id, f"input={request.input} output={output}")
        event = event_logger.emit(
            "agent_completed",
            agent_id=self.agent_id,
            task_id=request.task_id,
            payload={
                "output_preview": output[:200],
                "artifact_keys": sorted(artifacts.keys()),
            },
        )
        return AgentResponse(
            agent_id=self.agent_id,
            output=output,
            artifacts=artifacts,
            events=[event],
        )

    def _run(self, request: AgentRequest) -> str:
        output, _ = self._run_with_artifacts(request)
        return output

    def _run_with_artifacts(self, request: AgentRequest) -> tuple[str, dict[str, Any]]:
        total_started = perf_counter()
        timings: dict[str, float] = {}

        started = perf_counter()
        cleanup_stats = cleanup_deep_research_artifacts()
        timings["cleanup_seconds"] = round(perf_counter() - started, 3)

        if request.context.get("mode") == "group_chat":
            return (
                "deep_research 是长耗时研究流程。请单独使用 @deep_research 提交明确研究主题。",
                {"skipped": True, "reason": "batch_guard", "cleanup": cleanup_stats},
            )

        chapter14_path = Path(settings.chapter14_backend_path).resolve()
        if not chapter14_path.exists():
            return (
                f"chapter14 后端路径不存在，无法运行 deep_research：{chapter14_path}",
                {
                    "ready": False,
                    "chapter14_backend_path": str(chapter14_path),
                    "cleanup": cleanup_stats,
                },
            )

        if request.context.get("dry_run"):
            return (
                "deep_research 已接入 chapter14 后端路径，真实运行时会调用 chapter14 的 DeepResearchAgent。",
                {
                    "ready": True,
                    "chapter14_backend_path": str(chapter14_path),
                    "cleanup": cleanup_stats,
                },
            )

        started = perf_counter()
        DeepResearchAgent, Configuration = self._load_chapter14_types(chapter14_path)
        timings["load_chapter14_seconds"] = round(perf_counter() - started, 3)

        started = perf_counter()
        config = Configuration.from_env(overrides=self._chapter14_overrides())
        agent = DeepResearchAgent(config=config)
        timings["agent_init_seconds"] = round(perf_counter() - started, 3)

        started = perf_counter()
        result = agent.run(request.input)
        timings["agent_run_seconds"] = round(perf_counter() - started, 3)

        started = perf_counter()
        todo_items = [self._serialize_todo(item) for item in result.todo_items]
        report = result.report_markdown or result.running_summary or ""
        completed_items = [
            item for item in todo_items if item.get("status") == "completed" and item.get("summary")
        ]
        artifacts: dict[str, Any] = {
            "report_markdown": report,
            "todo_items": todo_items,
            "cleanup": cleanup_stats,
        }
        timings["postprocess_seconds"] = round(perf_counter() - started, 3)
        timings["total_seconds"] = round(perf_counter() - total_started, 3)
        artifacts["timings"] = timings
        if todo_items:
            artifacts["todo_count"] = len(todo_items)
            artifacts["completed_count"] = len(completed_items)

        if todo_items and not completed_items:
            output = (
                "搜索员没有拿到可用的搜索总结，因此未返回正式研究报告。\n"
                "可能原因：搜索后端无结果、网络 API 调用失败，或任务执行阶段没有产出摘要。\n"
                "请查看后端日志和 data/deep_research/runs 目录下的 task_* 文件。"
            )
            return output, artifacts

        output = report.strip()
        if not output:
            output = "deep_research 已完成，但没有生成报告正文。"

        return output, artifacts

    def _load_chapter14_types(self, chapter14_path: Path) -> tuple[type[Any], type[Any]]:
        path_text = str(chapter14_path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)

        # Chapter14 loads its own .env on import; reload chapter16 .env afterwards.
        agent_module = importlib.import_module("agent")
        config_module = importlib.import_module("config")
        if ENV_FILE.exists():
            load_dotenv(ENV_FILE, override=True)

        return agent_module.DeepResearchAgent, config_module.Configuration

    def _chapter14_overrides(self) -> dict[str, Any]:
        overrides: dict[str, Any] = {
            "notes_workspace": self._resolve_workspace(settings.notes_workspace),
            "run_workspace": self._resolve_workspace(settings.run_workspace),
        }

        optional_values = {
            "llm_provider": settings.llm_provider,
            "llm_model_id": settings.llm_model_id,
            "llm_api_key": settings.llm_api_key,
            "llm_base_url": settings.llm_base_url,
            "llm_timeout": settings.llm_timeout,
            "search_api": settings.search_api,
            "max_web_research_loops": settings.max_web_research_loops,
            "fetch_full_page": settings.fetch_full_page,
            "enable_notes": settings.enable_notes,
            "persist_runs": settings.persist_runs,
            "cleanup_intermediate_files": settings.cleanup_intermediate_files,
        }
        for key, value in optional_values.items():
            if value is not None:
                overrides[key] = value

        return overrides

    @staticmethod
    def _resolve_workspace(value: str) -> str:
        path = Path(value)
        if not path.is_absolute():
            path = ROOT_DIR / path
        path.mkdir(parents=True, exist_ok=True)
        return str(path.resolve())

    @staticmethod
    def _serialize_todo(item: Any) -> dict[str, Any]:
        return {
            "id": getattr(item, "id", None),
            "title": getattr(item, "title", ""),
            "intent": getattr(item, "intent", ""),
            "query": getattr(item, "query", ""),
            "status": getattr(item, "status", ""),
            "summary": getattr(item, "summary", None),
            "sources_summary": getattr(item, "sources_summary", None),
            "note_id": getattr(item, "note_id", None),
            "note_path": getattr(item, "note_path", None),
        }
