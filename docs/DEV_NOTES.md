# 开发备注（本机环境相关）

主 `README.md` 是给仓库读者看的；本文件记录只在**这台开发机**上成立的环境细节与踩坑，方便后续复现实验环境。这里不含任何凭证，路径用占位符表示。

## 已验证的环境组合

| 项 | 值 |
| --- | --- |
| Python | 3.13（Windows 侧） |
| PennyLane | 0.45.1 |
| torch / torchvision | 2.15.0 / 0.29.2（CPU 版） |
| numpy / scikit-learn | 2.5.4 / 1.x |
| 模拟器 | `default.qubit`，解析 `backprop`，无真机 |
| 显卡 | RTX 4060 Laptop 8GB（默认实验未使用；GPU 后端需 Linux） |

## 1. pip 源问题

全局 pip 配置指向清华镜像，但该环境里连 `torch` 都取不到，首次安装失败。改用阿里云镜像 + 官方 PyPI 兜底即可：

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

可用镜像（优先第一个）：

- `https://storage.googleapis.com/cvdf-datasets/mnist/`
- `https://ossci-datasets.s3.amazonaws.com/mnist/`

因此 `qml_lab/data.py` 增加了一个 `idx` 数据源，直接读本地 4 个 IDX 文件，完全离线且最快；`auto` 回退链现在是 **idx → torchvision → openml → synthetic**。正式实验用 `--data-source idx` 跑的是真实 MNIST，`synthetic` 只在冒烟测试中出现并会在 `meta.note` 里标注。

## 3. Windows 与 WSL 双副本

开发副本在 Windows 工作区，git 仓库在 WSL Ubuntu 的 home 目录下，两边靠复制同步。同步时要排除 `.venv/`、`results/`、`data/`（体积大且是本机产物）：

```bash
rsync -a --exclude '.venv/' --exclude 'results/' --exclude 'data/' \
    /mnt/c/Users/<WIN_USER>/WorkBuddy/quantum-machine-learning-code/ \
    /home/<WSL_USER>/quantum-machine-learning-code/quantum-mnist/
```

从 Windows 侧也可以通过 UNC 路径 `\\wsl$\Ubuntu\home\<WSL_USER>\...` 直接读写 WSL 文件（实测读写均正常），不必启动 WSL 进程。但如果要在 WSL **内部执行** Python / git 命令，仍需 `wsl.exe`；该程序若被安全策略的「程序黑名单」拦截，需要使用者手动在「安全中心 → 命令安全 → 程序黑名单」移除，这一步无法由自动化流程代劳。

另一个可行做法：不进 WSL，直接在 Windows 侧用 git 操作 UNC 路径下的仓库：

```bash
R='//wsl$/Ubuntu/home/<WSL_USER>/quantum-machine-learning-code/quantum-mnist'
git --git-dir="$R/.git" --work-tree="$R" status
```

## 4. 复现检查清单

```bash
python -m venv .venv
# 按第 1 节换源安装 requirements.txt
.venv/Scripts/python.exe tests/smoke_test.py                     # 或 .venv/bin/python
.venv/Scripts/python.exe experiments/01_quantum_mnist.py --quick
.venv/Scripts/python.exe experiments/02_quantum_llm.py --quick
```

冒烟测试全程离线、不下载 MNIST。若它失败，优先检查 PennyLane 版本：`qml.qnn.TorchLayer` 的 QNode 接口、batch broadcasting 与输出 dtype（float64 → float32）在 0.45 下已验证可用，升级大版本时需要重新确认这三点。
