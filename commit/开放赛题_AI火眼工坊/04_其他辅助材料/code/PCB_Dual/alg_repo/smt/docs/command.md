# 常用命令汇总

## 环境设置

### Windows (PowerShell)
```powershell
$env:PYTHONUTF8=1
$env:PYTHONPATH="src;contract_reference"
```

### Ubuntu
```bash
export PYTHONUTF8=1
export PYTHONPATH=src:contract_reference
```

> 在 Dev_solder_smt_Alg 目录下执行以下命令。

---

## 1. CLI 单图检测

### Windows
```cmd
python -m algorithms.solder_smt.composite_smt_all.solder_smt_all_alg "NG_image.png" -t "NG_image_OK.png" --pad "pad.json"
```

### Ubuntu
```bash
python -m algorithms.solder_smt.composite_smt_all.solder_smt_all_alg "NG_image.png" -t "NG_image_OK.png" --pad "pad.json"
```

### 加载params.json + 只检测虚焊
```bash
python -m algorithms.solder_smt.composite_smt_all.solder_smt_all_alg \
    "NG_image.png" -t "NG_image_OK.png" --pad "pad.json" \
    --params "params.json" --no-excess --no-insufficient --no-bridge --no-vis
```

### CLI 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `image` | 输入图片路径（必填） | - |
| `-t, --template` | 模板图片路径（*_OK.png） | 无 |
| `--pad` | pad.json 路径 | 自动查找图片同目录 |
| `--params` | params.json 路径 | 无 |
| `--no-excess` | 关闭多锡检测 | 开启 |
| `--no-insufficient` | 关闭少锡检测 | 开启 |
| `--no-bridge` | 关闭连锡检测 | 开启 |
| `--no-cold` | 关闭虚焊检测 | 开启 |
| `--no-vis` | 关闭可视化图（提速） | 开启 |

---

## 2. 启动 tuner 调参工具

### Windows
```cmd
set PYTHONPATH=src;contract_reference
python tuner\pcba_tuner_real.py
```

### Ubuntu
```bash
export PYTHONPATH=src:contract_reference
python tuner/pcba_tuner_real.py
```

---

## 3. 运行单元测试

### Windows
```cmd
set PYTHONUTF8=1
set PYTHONPATH=src;contract_reference
python -m pytest test_src/algorithms/solder_smt/solder_smt_all_alg_test.py -v
```

### Ubuntu
```bash
export PYTHONUTF8=1
export PYTHONPATH=src:contract_reference
python -m pytest test_src/algorithms/solder_smt/solder_smt_all_alg_test.py -v
```

---

## 4. C++ 加速模块编译

### Ubuntu
```bash
sudo apt-get install g++ python3-dev libopencv-dev
pip install pybind11>=2.10.0
rm -rf build _fast_core*.so
python csrc/setup.py build_ext --inplace --force
```

### 验证
```bash
python -c "from _fast_core import compute_diff_fast, detect_crack_hough_fast, detect_crack_batch_fast; print('C++ OK')"
```

### Windows
```cmd
pip install pybind11>=2.10.0
python csrc\setup.py build_ext --inplace --force
```

> 如果不编译，算法自动回退到纯Python，功能完全一致，只是稍慢。

---

## 5. 验证 C++ 模块状态

```bash
python -c "from _fast_core import compute_diff_fast, detect_crack_hough_fast, detect_crack_batch_fast; print('三个C++函数都OK')"
PYTHONPATH=src:contract_reference python -c "from algorithms.solder_smt.composite_smt_all._solder_core import _USE_CPP; print('_USE_CPP:', _USE_CPP)"
```

---

## 6. 测试集图片路径

```
test_images/
├── bridge_pair/         # 连锡测试
├── cold_solder_pair/    # 虚焊测试
├── excess_pair/         # 多锡测试
└── insufficient_pair/   # 少锡测试
```

每个pair目录包含：待检图(*.png) + 模板图(*_OK.png) + pad.json + params.json
