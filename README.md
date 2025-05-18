# RL_Learning_Practice
该仓库是用户用于学习强化学习的代码，代码参考为CSDN博主云端FFF和《动手学习强化学习》。
# 1.Cliff Walking Problem 🧗‍♂️

一个基于 **OpenAI Gym** 的可视化 Cliff-Walking 环境 + 经典表格型强化学习算法（SARSA / Expected SARSA）的教学 & 实验仓库。

* **环境完全自研**：支持任意网格尺寸、可自由切换 `human / rgb_array` 渲染模式。
* **实时可视化**：价值函数用颜色深浅渲染，贪心策略用绿虚线箭头渲染。
* **多种交互方式**：键盘手动控制、随机动作测试、脚本自动训练。
* **算法即插即用**：表格版 SARSA、Expected SARSA（支持 on-policy / off-policy + 经验回放）。

> 📚 代码风格与注释皆面向 RL 初学者，适合作为课程 / 读书会配套 Demo。

---

## 目录结构

```text
RL_Learning_Practice/
├── Cliff_Walking_Problem/
│   ├── CliffWalking/            # 环境包
│   │   ├── envs/
│   │   │   └── GridWorld.py     # CliffWalkingEnv 实现
│   │   └── wrappers/
│   │       └── Hashposotion.py  # 观测包装器：二维坐标 → 一维哈希
│   ├── EnvTest_ManualControl.py # 键盘控制 Demo
│   ├── EnvTest_RandomAction.py  # 随机动作 Demo
│   ├── RL_Solver_Class.py       # Solver 基类 + SARSA / Expected SARSA
│   ├── RL_Sarsa.py              # SARSA 训练脚本
│   ├── RL_ExpectedSarsa.py      # Expected SARSA 训练脚本
│   ├── *.ipynb                  # 教程 / 试验笔记本
└── README.md                    # 本文件
```

---

## 依赖环境

本仓库暂未提供 `requirements.txt`，请按下表手动安装**最小运行依赖**：

| 库          |   版本建议   | 说明                       |
| :--------- | :------: | :----------------------- |
| Python     |   ≥ 3.8  | 推荐使用 Conda 虚拟环境          |
| gym        | \~= 0.26 | `gym.utils.play` 仍在此版本保留 |
| numpy      |  ≥ 1.24  | 数值计算                     |
| pygame     |   ≥ 2.0  | 图形渲染                     |
| matplotlib |   ≥ 3.7  | 训练曲线绘制                   |
| tqdm       |  ≥ 4.66  | 进度条                      |

> 其他包（如 `jupyter`）仅在 Notebook 场景下需要，可按需安装。

### 一键安装

```bash
conda create -n cliffwalk python=3.8 -y
conda activate cliffwalk
pip install "gym~=0.26" pygame numpy matplotlib tqdm
```

---

## 快速上手

### 1️⃣ 键盘体验环境

```bash
python EnvTest_ManualControl.py
```

* 使用 **↑ ↓ ← →** 移动代理；
* 空格或其他键触发 `noop`（保持不动）；
* 终端实时打印奖励与信息。

### 2️⃣ 随机动作示例

```bash
python EnvTest_RandomAction.py
```

脚本会随机采样动作，并将随机生成的价值函数 & 策略渲染到窗口。

### 3️⃣ 训练 SARSA

```bash
python RL_Sarsa.py
```

训练过程中：

* Pygame 窗口实时显示 **价值函数 + 贪心策略**；
* 终端输出每 5 个 episode 的平均回报；
* 结束后自动绘制回报曲线。

### 4️⃣ 训练 Expected SARSA

```bash
python RL_ExpectedSarsa.py
```

支持 on-policy（默认）与 off-policy（传入 `batch_size>0`）两种模式，脚本顶部可修改超参数。

---

## 算法简介

| 算法             | 更新目标                                        | 特点            |
| :------------- | :------------------------------------------ | :------------ |
| SARSA          | $Q(s,a) ← Q(s,a)+α[r + γQ(s',a') − Q(s,a)]$ | on-policy、易实现 |
| Expected SARSA | $r + γ\mathbb{E}_{a'\simπ}[Q(s',a')]$       | 目标更稳定，可扩展经验回放 |

详见 [`RL_Solver_Class.py`](Cliff_Walking_Problem/RL_Solver_Class.py)。

---

## 可视化说明

* **颜色深浅**：状态价值 $V(s)$，由红浅 → 深表示价值越高；
* **绿虚线箭头**：当前贪心策略动作；
* **蓝色圆点**：智能体实时位置；
* `rgb_array` 模式下返回原始 RGB 帧，便于后续接入 CNN。

<p align="center">
  <img src="docs/demo.gif" width="600" alt="demo" />
</p>

---

## Roadmap 🚧

* [ ] Q-Learning / Double Q-Learning
* [ ] Dyna-Q
* [ ] 函数逼近（DQN / Sarsa(λ)）
* [ ] 自动生成 `requirements.txt`

---

## 参考资料

* Sutton & Barto. *Reinforcement Learning: An Introduction*
* [OpenAI Gym 文档](https://www.gymlibrary.dev/)

---

## License

MIT © 2025
