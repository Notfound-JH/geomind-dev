# 可选 cuRobo 源码溯源

上游 <https://github.com/NVlabs/curobo.git> 固定到 `ebb71702f3f70e767f40fd8e050674af0288abe8`（原 `v0.7.7-5-gebb7170`）。

`experimental/` 与 `former-editable/` 分别保存原工作站两套 checkout 的 tracked 修改和新增作者代码/配置。它们基于相同上游，但不能互相当作等价版本，也不要叠加到同一 checkout。

每个变体的 `full_source.patch` 包含已修改 demo、机器人配置与记录的新源码；`provenance.json` 列出固定 SHA、补丁哈希和新增配置路径。FR3 描述/mesh、non_shipping 等大资源放在单独资产包，不放进源码 Git；CUDA 安装 `.deb` 和编译产品不打包。

当前验收用 Python 3.11 / Isaac Sim 5.1 环境没有安装 nvidia-curobo，当前 27 类组装/抓放 pipeline 不依赖它。本目录保留完整自定义源以便恢复可选旧规划工具，不声称这些 demo 或 CUDA 扩展已在目标主机通过。

需要该工作流时，从固定上游新 clone，选择一个补丁 `git apply --check` 后应用，按包内配置恢复 FR3 资产；再按目标 CUDA/PyTorch 与上游说明在目标机编译和单独运行 GPU smoke。不要直接搬运原主机的 `.so`、Conda 环境或绝对路径配置。
