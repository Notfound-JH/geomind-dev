# 资产包构建算法与验证源

这些是本次交付实际采用的构建算法和验证脚本，用于审阅资产从哪里来、如何改写引用、如何校验。**正常安装 GEOMIND 不需要执行它们**；安装请按仓库主 README 解压资产 ZIP、运行 `scripts/configure_portable_assets.py`。

`source_manifest.json` 给出 12 个 Python 源文件的角色、大小和 SHA256。未包含原始资产、完整 inventory、Git 元数据、环境导出或凭据。`build_input_roots.example.json` 描述输入数据根的含义。构建快照保留了本次构建时的 `ROOT`、`OUT`、`BASES` 参数；从原始数据重新生成时，应改成自己的 checkout、GEOMIND 原始资产、GRScenes、NVIDIA 镜像和 MDL 缓存位置。这些构建时参数不是另一台主机运行 GEOMIND 的路径要求。

如果只是把已经解压且校验通过的资产目录重新打 ZIP，使用不依赖模拟器的入口即可：

```bash
python migration/asset_bundle/build_sources/repack_from_manifest.py \
  --assets-root /path/to/geomind-assets-v1 \
  --output /path/to/geomind-assets-repacked.zip
```

它逐文件核对内部 manifest，拒绝越界路径和覆盖已有 ZIP，使用 ZIP64，并另写 ZIP 的 SHA256。不同 Python/zlib 版本可能产生不同压缩字节；文件内容与 manifest 哈希必须相同。该入口已做合成数据往返验证。

从原始数据重建的主要步骤为：

1. `inventory_asset_closure_v4.py` 收集全部场景、目录资产、机器人、USD/MDL/纹理依赖，正确处理包含 `@` 的 USDA 转义路径。
2. `validate_xml_closure.py` 检查 URDF、Collada、OBJ/MTL 引用；`curate_asset_inventory.py` 排除嵌套 Git、凭据目录和无关缓存。
3. `build_asset_zip.py` 在副本中归一化引用；`extract_bulk_stage.py` 在不同前缀解压并核对全部 CRC/SHA256。原资产不改写。
4. `finalize_asset_stage.py` 合入经官方 commit/Git blob 核对的 FR3 补充网格；`final_asset_stage_corrections.py` 仅补回 MDL 明确声明的缩略图，并为迁移后的历史 Atlas 元数据添加 `portable_view_only` 和原始源哈希。
5. `validate_relocated_asset_stage.py` 对实际解压文件重新验证相对引用及 102 个场景层，并单列 SDK 提供的 MDL 标准模块与归档 Xacro/ROS/Gazebo 源模板要求。
6. `pack_final_asset_zip.py` 从最终不可变 stage 干净生成 ZIP；`verify_final_asset_zip.py` 读取每个成员验证 CRC、SHA256、唯一安全路径；`asset_security_scan.py` 不输出凭据值。

部分原始构建工具需要 IsaacSim 5.1 的 USD Python 绑定。仅激活环境不一定会直接暴露 `pxr`；应使用项目的 `_python_environment(..., usd_only=True)` 环境启动这些离线工具，不启动 Kit/GPU，也不另外安装可能与 SDK 冲突的 `usd-core`。原始目录和完整库存只在构建主机保存；交付包内部 manifest 保留资源级源哈希、官方 FR3 来源和转换记录。

历史 Atlas 的路径重绑视图用于 UI 展示，不是原始物理证据，也不产生新的任务通过结论。27 类任务的原始验证证据另行交付。FR3 的现成 URDF 网格闭包已补齐；可选 cuRobo 规划执行和官方 Xacro 的 ROS/Gazebo 再生成工具链不在本轮物理验证声明内。
