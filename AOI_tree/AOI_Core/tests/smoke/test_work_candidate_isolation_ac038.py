"""AC-038 工作/候选隔离守卫：生产入口不得回到即学改生产的 legacy 路径；主干只读共享。"""
import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROD_DIRS = ("backend/api", "ui", "backend/self_learning")


def _calls(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            yield name, node


class ProductionEntryGuardTest(unittest.TestCase):
    def test_no_legacy_online_from_production(self):
        print("[+] AC-038 生产入口不调 engine.submit_feedback / 不传 candidate_only=False", flush=True)
        bad = []
        for d in PROD_DIRS:
            for p in (ROOT / d).rglob("*.py"):
                tree = ast.parse(p.read_text(encoding="utf-8"))
                for name, node in _calls(tree):
                    kws = {k.arg: k.value for k in node.keywords}
                    for k, v in (("candidate_only", False), ("allow_legacy_online", True)):
                        if isinstance(kws.get(k), ast.Constant) and kws[k].value is v:
                            bad.append(f"{p.name}:{node.lineno} {k}={v}")
                    if name == "submit_feedback" and isinstance(node.func, ast.Attribute):
                        owner = node.func.value
                        src = ast.unparse(owner)
                        # UI 走 API 客户端；service 内 legacy 分支受 allow_legacy_online 守卫
                        if "engine" in src and p.name != "service.py":
                            bad.append(f"{p.name}:{node.lineno} {src}.submit_feedback")
        self.assertEqual(bad, [])

    def test_service_legacy_branch_guarded(self):
        print("[+] AC-038 service 默认 candidate_only=True 且 legacy 须显式放行", flush=True)
        import inspect
        from backend.self_learning import service
        sig = inspect.signature(service.submit_feedback_record)
        self.assertIs(sig.parameters["candidate_only"].default, True)
        self.assertIs(sig.parameters["allow_legacy_online"].default, False)
        with self.assertRaises(service.FeedbackConflict):
            service.submit_feedback_record(1, "false_positive", 0, candidate_only=False)


class SharedBackboneTest(unittest.TestCase):
    def test_backbone_shared_readonly(self):
        print("[+] AC-038 主干进程级只读共享：eval + requires_grad=False + 同一对象", flush=True)
        import torch
        from algo.backbone import dino
        try:
            a = dino.FrozenDINO(device="cpu")
        except Exception as e:  # 无随包权重的环境
            self.skipTest(f"dinov2 权重不可用: {e}")
        b = dino.FrozenDINO(device="cpu")
        self.assertIs(a.model, b.model)
        self.assertFalse(a.model.training)
        self.assertFalse(any(p.requires_grad for p in a.model.parameters()))
        self.assertIsInstance(a.model, torch.nn.Module)


if __name__ == "__main__":
    unittest.main()
