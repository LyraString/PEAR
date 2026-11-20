# PEAR 双 PiPER 实机联调交接

## 当前目标和状态

- PEAR 模型推理必须在远端 A100 工作站运行。现场 Linux 控制机连接双 PiPER 和相机，负责采集观测、坐标转换和动作执行。
- 已确认现场控制机能 SSH 到 A100。实机观测和动作链路尚待联调。
- A100 上的 PEAR 仓库是 `/data0/zhj/pear`，运行目录是 `/data0/zhj/pear_work`。
- 现场控制代码的副本在 A100 的 `/data0/zhj/collect_data/lerobot`。从这里查 PiPER 驱动和现场采集配置。

## A100 启动和验证

- 用 `examples/models/qwen3_8_27b_pear/deploy/run_server_a100.sh`。它使用现有 `$WORK_DIR/envs/verl`、本地 PEAR NVFP4 和 DFlash2 权重，沿用已验证的 TP=2、Marlin、BF16 和 `FULL_DECODE_ONLY`。
- 脚本绑定 `127.0.0.1:8000`，默认模型名 `pear`。启动前检查所选卡的空闲显存和其他用户的计算进程。40 GiB A100 每卡要求至少 34 GiB 空闲。
- 新脚本已通过 Bash 语法和本机 vLLM 0.28.0 参数检查。服务尚待空闲双卡上启动，131,072 token 的显存占用和现场多图请求待实测。
- 默认按 24 轮、72 张图和 131,072 token 规划。640×480 每帧实测 300 个视觉 token，加两个标记后三图共 906 token。再按每轮 256 token 的状态及格式、4,096 token 的输出，加 2,048 token 系统提示，共 128,240。4,096 是估算值，`xhigh` 的长输出或额外参考图会减少可用轮数。
- 历史图通过 UUID 引用，仍计入每次请求的图像数。额外参考图或更长任务需相应调整脚本的 `MAX_IMAGES`、`MAX_MODEL_LEN` 和站端上限。

空闲后在 A100 启动。`GPUS` 填 `nvidia-smi` 的物理卡编号，启动日志在 `$RUN/server.log`。

```bash
cd /data0/zhj/pear
WORK_DIR=/data0/zhj/pear_work GPUS=2,1 RUN=/data0/zhj/pear_work/runs/pear_server_a100 \
  bash examples/models/qwen3_8_27b_pear/deploy/run_server_a100.sh
# 另开终端确认 API 返回模型 pear
curl -fsS http://127.0.0.1:8000/v1/models
```

现场机通过 SSH 转发模型端口，替换 `A100_SSH_HOST` 为实际登录地址。

```bash
ssh -N -L 127.0.0.1:18000:127.0.0.1:8000 zhj@A100_SSH_HOST
# 另开终端验证模型 API
curl -fsS http://127.0.0.1:18000/v1/models
```

站端用 `SERVER=127.0.0.1:18000`、`NAME=pear`，在 `$STATION_CONFIG` 指向的 YAML 中合并以下配置。`run_station.sh` 会在此基础上设置客户端地址和模型名。

```yaml
models:
  served:
    client:
      context_limit: 131072
      max_images: 72
      max_output_tokens: null
```

`context_limit` 用于站端显示，vLLM 执行实际的 token 上限。首轮实机联调保留 `xhigh`，让每轮由剩余上下文决定输出预算。单轮硬截断可能切掉动作 JSON。

## PiPER 现场接口

- 现场代码已有 `bi_piper_follower`，底层用官方 `pyAgxArm` 和 SocketCAN。策略控制的是两支从臂。
- 查 `collect_data_piper.sh` 和 `scripts/piper_can_ports.sh`。采集配置使用 `v189` 固件配置、USB-CAN 序列号映射、左腕／右腕／顶视三路 RealSense。
- `src/lerobot/robots/bi_piper_follower/bi_piper_follower.py` 的 `get_observation()` 返回双臂状态和图像。两侧开启 `include_ee_pose` 后可读取 `left_ee_pose`、`right_ee_pose`。
- PiPER 的 `ee_pose` 是各自基座下的法兰反馈，XYZ 用米，RPY 用弧度。`send_action()` 接收双臂绝对关节目标和夹爪开口百分比，返回已下发的目标。
- PEAR 的双臂动作是共享坐标系下的末端增量，单位是厘米和弧度。参考 `pear/deploy/station/control/robots/yam/prompts/bimanual_pose.md` 的动作结构，按 PiPER 实际安装重写坐标和相机说明。
- 接动作需要标定左右基座之间的变换和法兰到夹爪的工具偏移，再经 PiPER 逆运动学转为关节目标。可评估复用 LeRobot 的 `src/lerobot/model/kinematics.py`，它需要对应 PiPER 的 URDF。

## 下一步

1. 现场先验证两支从臂反馈、三路图像和 SSH 转发。只读验证时用 `receive_only` 或 `enable_on_connect=false`。普通 follower 连接默认会使能电机。
2. 发两轮真实观测给 A100，先打印动作 JSON。第二轮验证历史图 UUID 引用，同时核对左右臂、轴方向、单位和夹爪字段。
3. 实测确认基座和工具坐标后再接本地动作执行。先验证单步动作及到位反馈，再跑完整任务。

## 已验证的仿真结果

- PEAR 官方 LIBERO 评测脚本 `examples/eval/eval_libero.sh` 已在双 A100 上跑通。
- 每个任务取前 10 个初始状态，共 400 条。NVFP4 + DFlash2、`xhigh`、61,440 token 响应预算，总成功率 67.0%。Spatial 88%、Object 93%、Goal 61%、Long 26%。
- 132 条失败中 120 条耗尽响应预算。汇总见 `/data0/zhj/pear_work/runs/libero_400.zGXj55/val_dumps/eval_0_summary.json`。
