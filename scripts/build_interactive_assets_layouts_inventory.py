"""Record candidate interactive assets and the actual layout code capabilities.

Read-only source audit. No source catalog rewrites, GPU launches or model calls.
"""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from geomind_lab.utils import asset_utils

OUT = ROOT/'outputs/interactive_assets_layouts_v1'
LABELS = {
    'blue_can':'蓝色咖啡罐','red_can':'红色番茄汤罐','low_can':'矮金枪鱼罐',
    'cube_can':'方形肉罐','yellow_bottle':'黄色芥末瓶','white_bottle':'白色清洁剂瓶',
    'red_box':'红色饼干盒','brown_box':'棕色布丁盒','rubiks_cube':'魔方',
    'glass_can':'玻璃罐','green_tin':'绿色罐','plate_1':'盘子1','plate_2':'盘子2',
    'glass_cup':'玻璃杯','banana':'香蕉','apple':'苹果','lime':'青柠',
    'pomegranate':'石榴','avocado':'牛油果','bowl_1':'碗1','bowl_2':'碗2',
    'basket_1':'篮子1','basket_2':'篮子2','basket_3':'篮子3','bin':'KLT周转箱',
    'cup_1':'杯子1','cup_2':'杯子2','pot_1':'盆状摆件1','pot_2':'盆状摆件2',
    'basket_4':'篮子4','box_1':'盒子1','box_2':'盒子2','box_3':'盒子3',
    'blue_tin':'蓝色罐（独立资产文件）','tin':'罐（独立资产文件）',
    'primitive_cuboid':'长方体/方块','primitive_sphere':'球体',
    'primitive_cylinder':'圆柱体','primitive_cone':'圆锥体',
}
POOL_ALL = {'blue_can','red_can','low_can','rubiks_cube'}
POOL_X = {'yellow_bottle','white_bottle','cube_can'}
SOURCE_FILES = [
    'scene_construction /arrange_object.py','scene_construction /arrange_object_new.py',
    'scene_construction /configs/object_config.py','scene_construction /configs/asset_config.py',
    'scene_construction /scene_utils/poisson_disk_disks.py',
    'tests/render_static_arrangement_scene.py','scripts/render_random_scene.py',
    'geomind_lab/tasks/manager_based/manipulation/geomind/layout_randomizer.py',
    'geomind_lab/tasks/manager_based/manipulation/geomind/task_specs.py',
]


def layout_rows():
    def row(code,layout,label,randomization,scope,limits,source,line):
        return dict(code=code,layout=layout,label_zh=label,randomization=randomization,
                    object_scope=scope,limits=limits,source=source,line=line)
    legacy='scene_construction /arrange_object.py'
    static='tests/render_static_arrangement_scene.py'
    hybrid='scripts/render_random_scene.py'
    current='geomind_lab/tasks/manager_based/manipulation/geomind/layout_randomizer.py'
    rows=[
        row('旧版物体摆放','Line','直线','资产抽样、顺序打乱、XY噪声、预设朝向和角度噪声','默认OBJECT_ALL；可显式传入USDAsset常量','根据桌子长边选择X/Y轴；显式interval覆盖自动间距',legacy,265),
        row('旧版物体摆放','Grid','二维网格','资产抽样、顺序打乱、XY/朝向噪声','默认OBJECT_ALL+OBJECT_X','按总数和桌子长边分配行列；可不满最后一行',legacy,278),
        row('旧版物体摆放','Tshaped','T形','按桌子长边在两个朝向间抽样；资产、顺序、XY/朝向噪声','默认OBJECT_ALL+OBJECT_X','底层有U/D/L/R四向，入口按长边选两向；坐标构造为5点；数量过多会越界；offset被重复叠加',legacy,291),
        row('旧版物体摆放','Ring','环形','资产、顺序、XY/朝向噪声','默认OBJECT_ALL+OBJECT_X','半径min(桌面两方向安全半径,0.2m)；不是任意物体碰撞规划',legacy,312),
        row('旧版物体摆放','TwoType','两条平行线/两组','两种资产抽样；各组数量；XY/朝向噪声','默认OBJECT_ALL+OBJECT_X；不打乱组顺序','应传两组数量；抽样用random.choices，可能抽到同一种物体',legacy,306),
        row('旧版物体摆放','Random','Poisson随机散点','中心坐标、资产、顺序、朝向','默认OBJECT_ALL','默认r=0.08m，最小中心距实际2r=0.16m；拥挤时采不满且上层未处理；默认自动间距分支关闭额外XY噪声',legacy,299),
        row('静态方块预览','line','直线','数量范围、间距比例噪声、独立XY噪声、seed','程序化彩色方块','支持X/Y轴；不代表YCB模型已接入此入口',static,507),
        row('静态方块预览','matrix','矩阵','数量范围模式随机选合法行列；间距/XY噪声','程序化彩色方块','仅2×2、2×3、3×2、3×3；数量范围模式不按任意min/max采样',static,522),
        row('静态方块预览','t_shape','T形','间距/XY噪声、seed','程序化彩色方块','实际固定5块，覆盖传入数量',static,538),
        row('静态方块预览','circle','圆环','数量范围、间距/XY噪声、seed','程序化彩色方块','可指定圆环半径',static,561),
        row('静态方块预览','basket_center','篮子中心布局','数量范围、间距/XY噪声、seed','程序化彩色方块+容器','按尺寸和间距生成篮内位置；并非任意资产容器放置保证',static,577),
        row('静态方块预览','stack_center','桌中心堆叠','数量范围、seed、顶层尺寸比例','程序化彩色方块','共用XY，跳过XY噪声；调整层尺寸及高度；稳定性需物理验证',static,597),
    ]
    for key,label in [('Line','直线'),('Grid','网格'),('Tshaped','T形'),('Random','随机散点'),('TwoType','两组平行线')]:
        limits='旧分支，不含Ring；没有按桌子尺寸自动适配的新接口'
        if key=='Grid':limits+='；硬编码WHITE_BOTTLE，覆盖传入资产'
        if key=='Random':limits+='；区域固定中心±0.2m；默认r=.05m，中心最小间距.10m'
        rows.append(row('旧new版本',key,label,'方向/间距/噪声等旧实现','USDAsset；Grid强制白瓶','不建议作为统一入口；'+limits,'scene_construction /arrange_object_new.py',220))
    for key,label in [('line','直线'),('grid','网格'),('t_shape','T形'),('l_shape','L形'),('circle','圆环'),('scatter','随机散点'),('stack','堆叠'),('container','篮内黄色瓶'),('auto','随机模式')]:
        limit='脚本依赖旧SiFERbench目录及siferbench模块，当前预期同级目录缺失；未执行'
        if key=='stack':limit+='；USD模式仅LOW_CAN，其余模式用方块'
        if key=='scatter':limit+='；自有分支均匀独立抽点，无Poisson最小间距保证'
        if key=='container':limit+='；专用篮子+黄色瓶分支'
        if key=='auto':limit+='；随机模式/结构，部分分支概率堆叠'
        rows.append(row('混合随机场景脚本',key,label,'seed；模式/资产/数量/颜色等随机化','USD资产、VisualCuboid外观块或混合；container为专用分支',limit+'；外观块不等同交互刚体，混合分支不保证整组统一结构',hybrid,615 if key!='container' else 362))
    for key,label in [('rank_line','保持一维顺序'),('midpoint','保持中点关系'),('point_symmetry','保持点对称'),('extension','保持等距延伸'),('square','保持正方形结构'),('trend','保持趋势延续'),('grid','保持行列网格'),('free','自由刚体组扰动'),('spatial_free','工作区内关系放置采样'),('frozen','相对几何冻结')]:
        rows.append(row('当前任务结构保持随机化',key,label,'jitter+整体shift；独立seed；拒绝采样','runtime_objects；按protocol和GT resolver选择策略','默认关闭；检查边界、近似足迹间距和派生GT点；frozen仅允许整体平移；不是通用抓放或无碰撞保证',current,1096))
    return rows


def random_rows():
    return [
        dict(code='旧版物体摆放',parameter='objects / object_num',default='按布局选择池；数量默认[4]',meaning='显式物体常量列表与每类数量；未提供objects时按池抽样',limit='默认只有7种基础资产；不是19种全自动采样'),
        dict(code='旧版物体摆放',parameter='noise / znoise',default='0.02m / π/24rad',meaning='独立XY偏移；预设朝向+角度噪声及随机翻转',limit='noise不能证明结构严格保持；Random仅自动interval分支将noise设0'),
        dict(code='旧版物体摆放',parameter='table_dims / interval / offset',default='[1,2]m / 自动 / 0',meaning='估算桌面安全区，15cm边缘、自动间距/半径、整体偏移',limit='不考虑真实网格足迹；center/offset和噪声后不做完整边界验收'),
        dict(code='旧版物体摆放',parameter='random seed',default='无专用seed参数',meaning='依赖Python random状态，可由外部random.seed控制',limit='不可将随机朝向与碰撞安全混同'),
        dict(code='静态方块预览',parameter='--num-objects-min/max',default='关闭，默认4块',meaning='数量范围抽样',limit='matrix仅四种行列组合；t_shape固定5块'),
        dict(code='静态方块预览',parameter='--spacing-jitter / --xy-jitter',default='0 / 0',meaning='间距乘U(1-j,1+j)，位置加U(-j,j)',limit='spacing-jitter限定0至0.5；堆叠不施加XY噪声'),
        dict(code='静态方块预览',parameter='--random-seed',default='未指定',meaning='复现数量、间距、位置等随机项',limit='必须同时保留配置、资产版本以复现渲染'),
        dict(code='当前任务随机化',parameter='GEOMIND_LAYOUT_JITTER_ENABLE / _JITTER_XY',default='0 / 0.02m',meaning='结构感知扰动或基于结构参数重建',limit='容器各部分作为刚体组；不同任务策略不同'),
        dict(code='当前任务随机化',parameter='GEOMIND_LAYOUT_SHIFT_ENABLE / _SHIFT_MAX',default='0 / 0.15m',meaning='全体随机对象整体平移',limit='受工作区与派生GT目标边界约束'),
        dict(code='当前任务随机化',parameter='GEOMIND_LAYOUT_JITTER_SEED / _SHIFT_SEED',default='未设时随机',meaning='两阶段独立种子',limit='固定种子也需保留相同输入布局'),
        dict(code='当前任务随机化',parameter='GEOMIND_LAYOUT_MIN_CLEARANCE',default='0.025m',meaning='基于物体近似平面足迹的间距检查',limit='原布局已近距离的物对允许保持近距离；不是网格接触检测'),
        dict(code='当前任务随机化',parameter='GEOMIND_LAYOUT_BOUNDS',default='0.15,0.68,-0.32,0.34',meaning='x_lo,x_hi,y_lo,y_hi，单位米',limit='需按新桌面/坐标系重设，不能直接沿用于所有102场景'),
        dict(code='当前任务随机化',parameter='GEOMIND_LAYOUT_MAX_ATTEMPTS',default='256',meaning='有界拒绝采样次数',limit='结构不匹配/采样失败时可能保留原布局或仅整体平移；以applied_layout_randomization记录为准'),
    ]


def build():
    OUT.mkdir(exist_ok=True,parents=True)
    table=asset_utils.assets_table
    assets=[]
    for category in ['catch_object','container']:
        for entry in table.get(category,[]):
            name=entry['name']
            if category=='container' and name.startswith('cabinet_'):continue
            path=Path(entry['usd_path']);resolved=path;status='配置路径存在'
            if not path.is_file() and '/Assets/ArchVis/' in str(path):
                resolved=Path('/home/elm/isaacsim_assets/Assets/Isaac/4.5/NVIDIA/Assets/ArchVis')/str(path).split('/Assets/ArchVis/',1)[1]
                status='旧配置路径失效；本机找到对应模型' if resolved.is_file() else '文件未找到'
            assets.append(dict(asset=name,label=LABELS[name],category='登记小物体' if category=='catch_object' else '登记桌面容器',
                catalog_registered=True,configured_path=str(path),usable_local_path=str(resolved),file_exists=resolved.is_file(),
                path_status=status,default_pool='OBJECT_ALL' if name in POOL_ALL else 'OBJECT_X/OBJECT_Y' if name in POOL_X else '不在旧版默认池',
                scale=entry.get('scale'),notes='仅文件可用性；未逐资产物理/抓取验证'))
    for name in ['cup_1','cup_2','pot_1','pot_2']:
        path=ROOT.parent/f'geomind_assets/layout/{name}.usda'
        assets.append(dict(asset=name,label=LABELS[name],category='原场景便携摆件',catalog_registered=False,
            configured_path=str(path),usable_local_path=str(path),file_exists=path.is_file(),path_status='配置常量及原场景中存在',
            default_pool='不在旧版默认池',scale=None,notes='原场景中可作为桌面小件；需补交互/刚体配置再使用'))
    for name in ['basket_4','box_1','box_2','box_3','blue_tin','tin']:
        path=ROOT.parent/f'geomind_assets/objects/{name}.usda'
        assets.append(dict(asset=name,label=LABELS[name],category='目录补充文件',catalog_registered=False,
            configured_path='',usable_local_path=str(path),file_exists=path.is_file(),path_status='本地文件存在，未在当前目录登记',
            default_pool='不在旧版默认池',scale=None,notes='按文件名标识；尺寸、碰撞、语义与资产重复性尚未核验'))
    for name in ['primitive_cuboid','primitive_sphere','primitive_cylinder','primitive_cone']:
        assets.append(dict(asset=name,label=LABELS[name],category='程序化几何',catalog_registered=False,
            configured_path='',usable_local_path='',file_exists=None,path_status='代码生成，无独立USD文件',
            default_pool='由runtime_objects或专用入口选择',scale=None,notes='代码中有生成支持；不代表所有形状都有抓取执行证据'))
    identifiers={a['asset'] for a in assets};counts=Counter();scene_counts=Counter()
    for report_path in (ROOT/'assets/scenes/tabletop_cleared_v1').glob('*/removal_report.json'):
        seen=set()
        for removed in json.loads(report_path.read_text())['removed']:
            for path in removed['items']:
                name=path.rsplit('/',1)[-1]
                match=next((a for a in sorted(identifiers,key=len,reverse=True) if name==a or re.fullmatch(re.escape(a)+r'_\d+',name)),None)
                if match:counts[match]+=1;seen.add(match)
        scene_counts.update(seen)
    for asset in assets:
        asset['instances_in_102_scenes']=counts[asset['asset']]
        asset['scenes_in_102_scenes']=scene_counts[asset['asset']]
        variants=[]
        if asset['asset']=='basket_1':
            variants=[str(p) for p in sorted((ROOT.parent/'geomind_assets/objects').glob('basket_1*.usda'))]
            asset['notes']+='；含with_collision/no_collision等版本，应显式选择碰撞版本'
        if asset['asset']=='bin':asset['notes']+='；源文件名含visual，碰撞支持需另核验'
        asset['variant_paths']=variants
    assert len(assets)==39 and sum(counts.values())==638
    assert all(a['file_exists'] for a in assets if a['file_exists'] is not None)
    data=dict(assets=assets,layouts=layout_rows(),randomization=random_rows(),
        scope='当前目录25类小物体/容器，原场景4类便携摆件，6类未登记文件，4类程序化几何。家具、robot、重复朝向变体不计入独立交互资产。',
        catalog_source=str(asset_utils.ASSET_CATALOG_PATH),
        verification='只读源码与文件存在核查；CPU layout_randomizer 单元测试3项通过；未运行GPU或逐资产物理验证。',
        source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in SOURCE_FILES})
    (OUT/'inventory.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
    for filename,rows in [('interactive_assets.csv',assets),('layout_support.csv',data['layouts']),('randomization.csv',data['randomization'])]:
        with (OUT/filename).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
            writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in rows)
    print(json.dumps(dict(asset_rows=len(assets),registered=25,old_paths_resolved=sum(a['path_status'].startswith('旧配置') for a in assets),
        default_pool_base_types=7,extra_files=6,portable_decor=4,primitives=4,layouts=len(data['layouts']),historical_instances=sum(counts.values())),ensure_ascii=False))

if __name__=='__main__':build()
