# 开发备注（quantum-mnist）

主 `README.md` 给仓库读者看；本文件记录只在开发机上成立的环境细节与踩坑，方便复现实验环境。路径用占位符表示，不含凭证。

## 已验证的环境组合

| 项 | 值 |
| --- | --- |
| Python | 3.13（Windows 侧） |
| PennyLane | 0.45.1 |
| torch / torchvision | 2.15.0 / 0.29.2（CPU 版） |
| numpy / scipy / scikit-learn | 2.x / 1.18.1 / 1.9.1 |
| 模拟器 | `default.qubit`，解析 `backprop`，无真机 |

## 1. pip 源问题

全局 pip 配置指向清华镜像，但该环境里连 `torch` 都取不到。改用阿里云镜像 + 官方 PyPI 兜底即可：

```bash
python -m pip install -i https://mirrors.aliyun.com/pypi/simple/ \
    --extra-index-url https://pypi.org/simple --timeout 60 --retries 3 \
    -r requirements.txt
```

`torch` 与 `torchvision` 版本强绑定，安装前可用 `pip install --dry-run torchvision` 确认不会改动已装 torch。

## 2. MNIST 下载卡住 → 离线 IDX 数据源

`torchvision` 自动下载时测试集 `t10k-images-idx3-ubyte.gz` 只下到 426KB 就卡在重试。解决办法是手动从镜像补齐 4 个 gzip 文件并解压，放到 `data/MNIST/raw/`：

| 文件（解压后） | gzip 原始大小 |
| --- | --- |
| `train-images-idx3-ubyte` | 9912422 |
| `train-labels-idx1-ubyte` | 28881 |
| `t10k-images-idx3-ubyte` | 1648877 |
| `t10k-labels-idx1-ubyte` | 4542 |

可用镜像：`https://storage.googleapis.com/cvdf-datasets/mnist/`、`https://ossci-datasets.s3.amazonaws.com/mnist/`。

因此 `qml_lab/data.py` 增加了 `idx` 数据源，直接读本地 4 个 IDX 文件，完全离线且最快；`auto` 回退链是 **idx → torchvision → openml → synthetic**。正式实验用 `--data-source idx` 跑的是真实 MNIST，`synthetic` 只在冒烟测试中出现并会在 `meta.note` 里标注。

## 3. Windows 与 WSL 双副本

开发副本在 Windows 工作区，git 仓库在 WSL Ubuntu 的 home 目录，靠复制同步。同步时排除 `.venv/`、`results/`、`data/`：

```bash
rsync -a --exclude '.venv/' --exclude 'results/' --exclude 'data/' \
    /mnt/c/Users/<WIN_USER>/WorkBuddy/quantum-machine-learning-code/<repo>/ \
    /home/<WSL_USER>/quantum-machine-learning-code/<repo>/
```

从 Windows 侧也可以通过 UNC 路径 `\\wsl$\Ubuntu\home\<WSL_USER>\...` 直接读写 WSL 文件（实测读写正常），不必启动 WSL 进程。但要在 WSL **内部执行**命令仍需 `wsl.exe`；该程序若被安全策略「程序黑名单」拦截，需使用者手动在「安全中心 → 命令安全 → 程序黑名单」移除。

不进 WSL 也能提交：直接在 Windows 侧用 git 指向 UNC 下的仓库：

```bash
R='//wsl$/Ubuntu/home/<WSL_USER>/quantum-machine-learning-code/quantum-mnist'
git --git-dir="$R/.git" --work-tree="$R" status
```

## 4. 复现检查清单

```bash
python -m venv .venv
# 按第 1 节换源安装 requirements.txt
.venv/Scripts/python.exe tests/smoke_test.py                       # 或 .venv/bin/python
.venv/Scripts/python.exe -m pytest tests/smoke_test.py
.venv/Scripts/python.exe experiments/train_mnist.py --quick
.venv/Scripts/python.exe experiments/resolve_open_issues.py --quick
.venv/Scripts/python.exe experiments/train_mnist.py --data-source idx --seed 0
```

冒烟测试全程离线、不下载 MNIST。若它失败，优先检查 PennyLane 版本：`qml.qnn.TorchLayer` 的 QNode 接口、batch broadcasting 与输出 dtype（float64 → float32）在 0.45 下已验证可用，升级大版本时需重新确认这三点。
