# 插件焊点缺陷检测

标准图对比 + 偏移校准 + HSV 分割 + 规则判定。输入为焊点 ROI，输出 **OK / NG / 待复检**，NG 时框出缺陷。

五类缺陷：**孔洞(12)**、**少锡(13)**、**多锡(14)**、**连锡(15)**、**不出脚(16)**。  
自动锡面或 1 个人工框 → 仅盘内四种；≥2 个人工焊点框 → 仅连锡（`pad_ring` 主焊盘算位移，多框共用）。

## 架构

```
run.py / 算法控件
  → 模板预处理 (锡面 Mask 缓存)
  → inspect():
       配准 (pad_ring；多框时主焊盘 + 位移共用)
       → 覆盖率早停 (可选 enable_review)
       → 孔洞分割 ∥ 锡面分割 → 特征
       → 少锡(13) ∥ 不出脚(16) → 多锡(14)
       → 连锡(15)（≥2 pad）
       → 交叉抑制 + enabled_defects → OK/NG/待复检
```

- `solder_mode`：`ellipse` / `contour` / `roi`（人工框需 `solder_roi`）
- `roi`：矩形 `{"shape":"rect","x","y","w","h"}` 或圆形 `{"shape":"circle","cx","cy","r"}`
- 配准偏移超限或覆盖率不足 → **待复检**（可用 `enable_review=false` 关闭）
- 交叉抑制：不出脚 ↔ 露铜型少锡互相抑制；多锡命中时不再报不出脚

## 输入 / 输出

```
input/
├── standard_roi/templ_N.png[+.json]   # 标准图及锡面标注
└── roi/<n>-*.png                      # 测试图（前缀 n → templ_n）

outputs/
├── OK/  NG/  REVIEW/
```

## 使用

```bash
pip install -r requirements.txt
python run.py
python run.py --config config.yaml
python run.py --input input --out outputs
```

根目录存在 `config.yaml` 时自动加载。常用调参见该文件中的 `align` / `edge_align` / `coverage` / `void_*` / `insuf_*` / `enabled_defects`。

## 目录

```
src/algorithms/pcb_through_hole/   # 算法入口 + 检测引擎（交付）
src/resources/config/              # 实例配置 + 算法池注册示例
test_src/                          # 单元测试
contract_reference/                # 只读契约参考（pytest 用，勿拷入宿主）
docs/                              # 算法控件接入说明
run.py / config.yaml / input/      # 本地 CLI 自测
```

接入宿主算法池：见 `docs/插件焊点接入说明.md`。

## 自测

工作目录为 `detect/`：

```bash
set PYTHONUTF8=1   # Windows
python -m pytest test_src/ -v
```
