"""Index the cleared scene library using visual environment names.

Names identify environment configurations; original numeric IDs identify instances.
No USDA file is renamed or modified.
"""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/scene_index_v1'
SOURCE = ROOT / 'assets/scenes/tabletop_cleared_v1'

TABLES = {
    'table_1': ('white_marble_oval', '白色大理石纹椭圆桌'),
    'table_2': ('white_rectangular', '白色长方桌'),
    'table_3': ('light_wood_rectangular', '浅木色长方桌'),
    'table_4': ('dark_wood_rectangular', '深木色长方桌'),
    'table_5': ('dark_wood_rectangular', '深木色长方桌'),
    'table_6': ('black_rectangular', '黑色长方桌'),
    'table_7': ('blue_gray_rectangular', '蓝灰色长方桌'),
    'table_8': ('black_rectangular', '黑色长方桌'),
    'table_9': ('black_glass_round', '黑色玻璃圆桌'),
    'table_10': ('white_oval', '白色椭圆桌'),
    'table_11': ('white_ornate_rectangular', '白色雕饰长方桌'),
    'table_12': ('gray_marble_rectangular', '灰色大理石纹长方桌'),
    'table_13': ('clear_glass_round', '透明玻璃圆桌'),
    'table_14': ('gray_drawer_rectangular', '灰色带抽屉长方桌'),
    'table_15': ('black_rectangular', '黑色长方桌'),
    'table_16': ('light_wood_rectangular', '浅木色长方桌'),
}
GROUNDS = {
    'wood_ground': ('warm_brown', '暖棕色'),
    'wood_ground_2': ('reddish_brown', '红棕色'),
    'wood_ground_3': ('dark_brown', '深棕色'),
    'wood_ground_4': ('light_brown', '浅棕色'),
}

# Furniture descriptors are used only where one base name maps to distinct
# retained environment configurations. Counts refer to actual kept USD assets.
def furniture_discriminator(row):
    table, furniture = row['table_asset'], row['furniture_assets']
    if table == 'table_3': return 'black_office_chair', '黑色办公椅'
    if table == 'table_16': return 'orange_chair', '橙色帆布椅'
    if table == 'table_4':
        return ('houndstooth_armchairs', '两把千鸟格扶手椅') if 'chair_4_1' in furniture else ('mixed_armchairs', '千鸟格与米色扶手椅组合')
    if table == 'table_5': return 'gray_cabinet', '灰木边柜'
    if table == 'table_6': return 'white_corner_sofa', '白色转角沙发'
    if table == 'table_8': return 'white_sideboard', '白色边柜'
    if table == 'table_15':
        return ('curved_sofa', '弧形沙发') if 'sofa_5' in furniture else ('wood_trim_sofa', '木饰边转角沙发')
    if table == 'table_10':
        return ('orange_chair', '橙色帆布椅') if 'chair_7' in furniture else ('cushioned_bench', '白色软垫长坐凳')
    if table == 'table_12':
        return ('two_orange_chairs', '两把橙色帆布椅') if 'chair_7_1' in furniture else ('one_orange_chair', '一把橙色帆布椅')
    raise ValueError(f'Missing furniture descriptor: {row}')


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    inventory_path = ROOT / 'outputs/scene_atlas_cleared_v1/inventory.json'
    inventory = json.loads(inventory_path.read_text())
    rows = []
    for scene in inventory:
        n = scene['id']
        report_path = SOURCE / str(n) / 'removal_report.json'
        report = json.loads(report_path.read_text())
        removed = [x['path'] for x in report['removed']]
        def kept(path):
            return not any(path == p or path.startswith(p + '/') for p in removed)
        names = []
        for item in scene['children']:
            if item['name'] in ['function', 'decoration']:
                names.extend(c for c in item['children'] if kept(item['path'] + '/' + c))
            elif kept(item['path']):
                names.append(item['name'])
        table, = [x for x in names if re.fullmatch(r'table_\d+', x)]
        ground, = [x for x in names if x.startswith('wood_ground')]
        furniture = sorted(x for x in names if x not in [table, ground] and not x.startswith(('Wall', 'Physics', 'Looks', 'Dome', 'Tabletop')))
        feature, feature_zh = TABLES[table]
        ground_color, ground_zh = GROUNDS[ground]
        signature = (table, ground, tuple(furniture))
        row = dict(scene_id=n, room='livingroom', table_feature=feature,
                   table_feature_zh=feature_zh, ground_color=ground_color,
                   ground_color_zh=ground_zh, table_asset=table,
                   ground_asset=ground, furniture_assets=furniture,
                   base_name=f'livingroom_{feature}_{ground_color}',
                   furniture_suffix='', furniture_suffix_zh='',
                   cleared_usda=str(Path(report['output']).relative_to(ROOT)),
                   preview_url=f'http://127.0.0.1:8780/#scene-{n}',
                   signature=signature)
        rows.append(row)
    groups = defaultdict(list)
    for row in rows: groups[row['signature']].append(row)
    base_groups = defaultdict(set)
    for signature, members in groups.items(): base_groups[members[0]['base_name']].add(signature)
    canonical_names = set()
    group_rows = []
    for i, (signature, members) in enumerate(groups.items(), 1):
        first = members[0]
        if len(base_groups[first['base_name']]) > 1:
            suffix, suffix_zh = furniture_discriminator(first)
        else: suffix, suffix_zh = '', ''
        name = first['base_name'] + ('_' + suffix if suffix else '')
        assert name not in canonical_names, f'Unresolved environment name collision: {name}'
        canonical_names.add(name)
        for row in members:
            row.update(scene_name=name, environment_group=i, group_size=len(members),
                       furniture_suffix=suffix, furniture_suffix_zh=suffix_zh)
        group_rows.append(dict(environment_group=i, scene_name=name,
                               scene_ids=[r['scene_id'] for r in members],
                               group_size=len(members), table_feature=first['table_feature'],
                               table_feature_zh=first['table_feature_zh'],
                               ground_color=first['ground_color'], ground_color_zh=first['ground_color_zh'],
                               furniture_suffix=suffix, furniture_suffix_zh=suffix_zh,
                               table_asset=first['table_asset'],ground_asset=first['ground_asset'],
                               furniture_assets=first['furniture_assets']))
    rows.sort(key=lambda r: r['scene_id'])
    for row in rows: del row['signature']
    assert len(rows) == 102 and len(groups) == len(canonical_names) == 35
    assert sum(g['group_size'] for g in group_rows) == 102
    for row in rows: assert (ROOT / row['cleared_usda']).is_file()
    output = dict(schema_version=1, naming_pattern='livingroom_tablefeature_groundcolor[_furniture]',
                  naming_scope='environment_configuration', duplicate_instance_policy='share_name_keep_scene_id',
                  grouping_basis='Retained table asset, ground asset, furniture/decor asset identities and counts. Small pose variations and raw USD byte equality are not deduplicated.',
                  visual_basis='Table appearance and floor colour reviewed using material-corrected 768x768 renders. Marble describes visible pattern, not a physical material assay.',
                  source_inventory=str(inventory_path.relative_to(ROOT)),
                  source_inventory_sha256=hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
                  scenes=rows, environments=group_rows,
                  table_dictionary={k:{'token':v[0],'label_zh':v[1]} for k,v in TABLES.items()},
                  ground_dictionary={k:{'token':v[0],'label_zh':v[1]} for k,v in GROUNDS.items()})
    (OUT / 'scene_index.json').write_text(json.dumps(output, ensure_ascii=False, indent=2))
    columns = ['scene_id', 'scene_name', 'environment_group', 'table_feature', 'ground_color',
               'furniture_suffix', 'group_size', 'table_asset', 'ground_asset', 'furniture_assets', 'cleared_usda', 'preview_url']
    with (OUT / 'scene_index.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns);writer.writeheader()
        for row in rows:
            result = {k:row[k] for k in columns};result['furniture_assets']=';'.join(result['furniture_assets']);writer.writerow(result)
    (OUT / 'validation.json').write_text(json.dumps(dict(instances=102, environment_groups=35,
        unique_environment_names=35, furniture_disambiguated_groups=sum(bool(g['furniture_suffix']) for g in group_rows),
        shared_name_groups=sum(g['group_size']>1 for g in group_rows), all_paths_exist=True,
        source_files_renamed=False), indent=2))
    print((OUT/'validation.json').read_text())
    for g in group_rows:
        if g['furniture_suffix']: print(g['scene_name'],g['scene_ids'])

if __name__ == '__main__': build()
