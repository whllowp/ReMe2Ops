"""Edge-side dream-unit push: send extracted units to the cloud for sedimentation."""

from ...components import R
from ..base_step import BaseStep
from ..evolve.dream.utils import state_from_context, store_state
from .cloud import CloudClient, cloud_settings


def unit_paths(units: list[dict]) -> list[str]:
    """Sorted union of the source paths behind a unit list."""
    return sorted({str(path) for unit in units for path in unit.get("paths") or []})


@R.register("ops_push_step")
class OpsPushStep(BaseStep):
    """Push freshly extracted dream units to the cloud ``cloud_ingest`` job.

    Runs between ``dream_extract_step`` and ``dream_finish_step``: units whose
    push (or cloud-side integration) failed get their source paths appended to
    ``state.failed_paths``, so the finish step defers them and the next run
    re-extracts and re-pushes. Re-pushed units are safe — cloud integration
    deduplicates semantically via node_search.
    """

    async def execute(self):
        assert self.context is not None
        state = state_from_context(self)
        if not state.units:
            self.context.response.answer = "No dream units to push"
            self.context.response.metadata["push"] = {"units": 0, "delivered": 0, "failed_units": 0}
            return self.context.response

        host, port, timeout = cloud_settings(self)
        device = str(self.context.get("source_device") or self.kwargs.get("source_device") or "edge")
        payload = {
            "units": [
                {
                    "name": unit.get("name"),
                    "bucket": unit.get("bucket"),
                    "summary": unit.get("summary"),
                    "paths": unit.get("paths") or [],
                }
                for unit in state.units
            ],
            "source_device": device,
            "date": state.date,
        }

        def defer_all(reason: str) -> None:
            for path in unit_paths(state.units):
                if path not in state.failed_paths:
                    state.failed_paths.append(path)
            state.errors.append(reason)
            store_state(self, state)
            self.context.response.success = False
            self.context.response.answer = f"Cloud push failed: {reason}"
            self.context.response.metadata["push"] = {
                "units": len(state.units),
                "delivered": 0,
                "failed_units": len(state.units),
                "cloud": f"{host}:{port}",
            }

        try:
            async with CloudClient(host, port, timeout) as cloud:
                data = await cloud.post("cloud_ingest", payload)
        except Exception as exc:  # noqa: BLE001
            defer_all(f"{type(exc).__name__}: {exc}")
            return self.context.response

        if not data.get("success", False):
            defer_all(str(data.get("answer") or "cloud_ingest reported failure"))
            return self.context.response

        cloud_state = (data.get("metadata") or {}).get("dream") or {}
        state.integrate_results = list(cloud_state.get("integrate_results") or [])
        state.nodes_created = list(cloud_state.get("nodes_created") or [])
        state.nodes_updated = list(cloud_state.get("nodes_updated") or [])
        failed_units = list(cloud_state.get("failed_units") or [])
        if failed_units:
            # Cloud accepted the push but could not integrate some units; defer
            # their source paths so the next run re-extracts and re-pushes them.
            for path in unit_paths(failed_units):
                if path not in state.failed_paths:
                    state.failed_paths.append(path)
        store_state(self, state)

        delivered = len(state.units) - len(failed_units)
        self.context.response.answer = (
            f"Pushed {len(state.units)} unit(s) to cloud {host}:{port}: "
            f"{delivered} integrated, {len(failed_units)} deferred for retry"
        )
        self.context.response.metadata["push"] = {
            "units": len(state.units),
            "delivered": delivered,
            "failed_units": len(failed_units),
            "created": state.nodes_created,
            "updated": state.nodes_updated,
            "cloud": f"{host}:{port}",
        }
        return self.context.response
