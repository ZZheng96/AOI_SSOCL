"""数据适配器公共约定：所有适配器返回 bundle 字典，字段见 isolation.guard_bundle。"""
from dataclasses import dataclass, field


@dataclass
class Bundle:
    name: str
    init_normal: list          # train/good，训练用
    init_defect: list          # ≤30，协议内缺陷，训练用
    val: list = field(default_factory=list)    # [(path, label)]
    test: list = field(default_factory=list)   # [(path, label)]

    def as_dict(self):
        return {"init_normal": self.init_normal, "init_defect": self.init_defect,
                "val": self.val, "test": self.test}
