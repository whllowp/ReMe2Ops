"""End-to-end ops cloud-edge memory collaboration demo.

Drives a running edge instance (ops_edge config) and cloud instance (ops_cloud
config) through the full loop:

  1. seed     - record an incident conversation into edge short-term memory
  2. search   - tiered ops_search (muscle reflex -> daily -> cloud fallback)
  3. push     - distill daily notes into units and push them to the cloud
  4. feedback - record solved/failed usage outcomes on the edge
  5. promote  - usage-qualified SOPs move into the cloud muscle bucket
  6. sync     - edge mirrors muscle SOPs locally and pushes usage deltas
  7. reflex   - the incident query now hits the local muscle tier instantly

Stages 1/3/4(integrate-side) need an LLM configured on the instances
(``LLM_API_KEY`` etc.); stages 2/4/6/7 are pure retrieval/storage and work
without one. Run ``python scripts/ops_demo.py --help`` for options.
"""

import argparse
import json
import sys
from dataclasses import dataclass

import httpx

EDGE = "http://127.0.0.1:2333"
CLOUD = "http://127.0.0.1:2444"

INCIDENT_SESSION = "ops-demo-disk-full-001"
INCIDENT_QUERY = "服务器磁盘空间不足告警怎么处理"


def post(client: httpx.Client, base: str, action: str, payload: dict) -> dict:
    """POST a job and return the Response JSON, aborting the demo on failure."""
    resp = client.post(f"{base}/{action}", json=payload, timeout=300.0)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success", False):
        print(f"  !! {action} returned success=false: {data.get('answer')}")
    return data


def health(client: httpx.Client, base: str, label: str) -> None:
    data = post(client, base, "health_check", {})
    print(f"  [{label}] {data.get('answer', 'ok')}")


@dataclass
class Stage:
    name: str
    title: str
    needs_llm: bool = False


STAGES = [
    Stage("seed", "写入短期记忆：事故会话 → 边端 daily 工作记忆卡", needs_llm=True),
    Stage("search", "分层检索：肌肉反射 → 短期 → 云端回源"),
    Stage("push", "夜间提炼：dream 提取单元并推送云端沉淀", needs_llm=True),
    Stage("feedback", "使用反馈：记录 solved/failed（最优解的依据）"),
    Stage("judge", "决策模型快判：一次成功使用即判定是否固化（需决策模型）"),
    Stage("promote", "知识固化：达标 SOP 晋升 muscle 桶", needs_llm=True),
    Stage("sync", "肌肉下发：边端镜像 muscle + 上行 usage"),
    Stage("reflex", "反射命中：同一查询直接命中本地肌肉记忆"),
]


def stage_seed(client: httpx.Client, state: dict) -> None:
    messages = [
        {"role": "user", "name": "oncall", "content": [
            {"type": "text", "text": "生产服务器 disk-util 大于 90% 告警，/var/log 占了 80G，怎么处理？"},
        ]},
        {"role": "assistant", "name": "ops-agent", "content": [
            {"type": "text", "text": (
                "处理过程：1) du -sh /var/log/* 定位大文件；2) 确认 journalctl --vacuum-size=2G 清理 "
                "systemd journal；3) 对应用旧日志执行 logrotate 强制轮转；4) 观察告警恢复后，在 "
                "/etc/logrotate.d/ 固化按天轮转策略，防止复发。"
            )},
        ]},
    ]
    data = post(client, EDGE, "auto_memory", {
        "messages": messages,
        "session_id": INCIDENT_SESSION,
        "request_id": f"{INCIDENT_SESSION}-1",
        "memory_hint": "运维事故处理：磁盘空间不足的处理步骤",
    })
    print(f"  daily 卡片: {data.get('metadata', {}).get('changes', data.get('answer'))}")


def stage_search(client: httpx.Client, state: dict) -> dict:
    data = post(client, EDGE, "ops_search", {"query": INCIDENT_QUERY, "limit": 5})
    for item in data.get("metadata", {}).get("results", []):
        tiers = "+".join(item.get("tiers", []))
        print(f"  [{item.get('score', 0):.4f}] {item['path']} ({tiers})")
    return data


def stage_push(client: httpx.Client, state: dict) -> None:
    data = post(client, EDGE, "ops_dream_push", {"scan_days": 2, "max_units": 5, "source_device": "edge-demo"})
    push = data.get("metadata", {}).get("push", {})
    print(f"  {data.get('answer')}")
    created = push.get("created", [])
    for path in created:
        print(f"  云端新建: {path}")
    for path in push.get("updated", []):
        print(f"  云端更新: {path}")
    if created:
        state["promote_path"] = created[0]
        print(f"  后续将固化: {state['promote_path']}")


def stage_feedback(client: httpx.Client, state: dict) -> None:
    path = state.get("promote_path")
    if not path:
        print("  （跳过：没有可固化的 SOP 路径）")
        return
    # Simulate repeated successful use of the SOP across incidents.
    data = {}
    for i in range(6):
        data = post(client, EDGE, "ops_feedback", {
            "path": path,
            "outcome": "solved" if i < 5 else "failed",
            "query": INCIDENT_QUERY,
            "session_id": f"{INCIDENT_SESSION}-{i}",
        })
    print(f"  {path} stats={data.get('metadata', {}).get('path_stats')}")


def stage_judge(client: httpx.Client, state: dict) -> None:
    path = state.get("promote_path")
    if not path:
        print("  （跳过：没有可判定的 SOP 路径）")
        return
    stage_sync(client, state)  # usage 必须先上行，云端才有判定证据
    data = post(client, CLOUD, "ops_muscle_judge", {"paths": [path], "auto_promote": True})
    judge = data.get("metadata", {}).get("judge", {})
    for verdict_path, verdict in (judge.get("verdicts") or {}).items():
        if verdict.get("judged"):
            print(f"  {verdict_path}: muscle_worthy={verdict.get('muscle_worthy')} "
                  f"evidence={verdict.get('evidence')} promotable={verdict.get('promotable')}")
        else:
            print(f"  {verdict_path}: 未判定（{verdict.get('reason')}）——继续走 usage 慢路径")
    if judge.get("promoted"):
        state["auto_promoted"] = True
        state["promote_path"] = judge["promoted"][0]["path"]
        print(f"  ⚡ 决策模型即时固化: {state['promote_path']}")


def stage_promote(client: httpx.Client, state: dict) -> None:
    path = state.get("promote_path")
    if not path:
        print("  （跳过：没有可固化的 SOP 路径）")
        return
    if state.get("auto_promoted"):
        print("  （决策模型已即时固化，跳过手动晋升）")
        return
    data = post(client, CLOUD, "ops_muscle_promote", {"path": path})
    print(f"  {data.get('answer')}")
    if data.get("success", False):
        state["promote_path"] = data.get("metadata", {}).get("promoted", {}).get("to", path)


def stage_sync(client: httpx.Client, state: dict) -> None:
    data = post(client, EDGE, "ops_muscle_sync", {})
    meta = data.get("metadata", {})
    print(f"  {data.get('answer')}")
    for path in meta.get("written", []):
        print(f"  已镜像: {path}")


def stage_reflex(client: httpx.Client, state: dict) -> None:
    data = stage_search(client, state)
    results = data.get("metadata", {}).get("results", [])
    if results and "muscle" in results[0].get("tiers", []):
        print("  ✅ 条件反射命中：最优解已在本地下发，无需访问云端")
    else:
        print("  （尚未形成肌肉记忆命中；继续积累 usage 后重试 promote/sync）")


STAGE_FUNCS = {
    "seed": stage_seed,
    "search": stage_search,
    "push": stage_push,
    "feedback": stage_feedback,
    "judge": stage_judge,
    "promote": stage_promote,
    "sync": stage_sync,
    "reflex": stage_reflex,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge", default=EDGE, help="edge instance base URL")
    parser.add_argument("--cloud", default=CLOUD, help="cloud instance base URL")
    parser.add_argument("--stages", default=",".join(s.name for s in STAGES), help="comma-separated stage names")
    parser.add_argument("--force", action="store_true", help="run LLM stages even when the cloud looks unreachable")
    args = parser.parse_args()

    with httpx.Client() as client:
        print("== 健康检查 ==")
        health(client, args.edge, "edge")
        cloud_up = True
        try:
            health(client, args.cloud, "cloud")
        except Exception as exc:  # noqa: BLE001
            cloud_up = False
            print(f"  [cloud] 不可达: {exc}（LLM 阶段将被跳过）")

        demo_state: dict = {}
        selected = [s for s in STAGES if s.name in set(args.stages.split(","))]
        print()
        for stage in selected:
            print(f"== {stage.name}: {stage.title} ==")
            if stage.needs_llm and not cloud_up and not args.force:
                print("  （跳过：需要云端/LLM 可用）")
                continue
            try:
                STAGE_FUNCS[stage.name](client, demo_state)
            except Exception as exc:  # noqa: BLE001
                print(f"  !! 阶段失败: {type(exc).__name__}: {exc}")
                return 1
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
