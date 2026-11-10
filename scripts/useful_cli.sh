#文件路径导入
export GEOMIND_ASSETS_ROOT="${GEOMIND_ASSETS_ROOT:-/home/elm/projects/geomind_assets}"

#查找文件
rg --files "$GEOMIND_ASSETS_ROOT" 2>/dev/null | rg '(^|/)assets_catalog_with_metadata\.json$'