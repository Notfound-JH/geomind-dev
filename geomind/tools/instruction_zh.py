"""Translate recorded native task templates for display without changing evaluation evidence."""
import re

LABELS = {
    'the coffee can': '咖啡罐', 'the meat tin': '肉罐',
    'the yellow mustard bottle': '黄色芥末瓶',
    'the red endpoint': '红色端点物体', 'the blue endpoint': '蓝色端点物体',
    'the green point': '绿色参考物体',
    'the red block': '红色方块', 'the blue block': '蓝色方块',
    'the green block': '绿色方块', 'the yellow block': '黄色方块',
}


def translate_instruction(text):
    """Use the recorded English text, never today's potentially changed layout or GT."""
    suffix = " Each object's reference point is its configured rigid-body origin."
    origin = text.endswith(suffix)
    source = text[:-len(suffix)] if origin else text
    point = '参考点' if origin else '中心'
    label = lambda value: LABELS.get(value, value)
    match = re.fullmatch(
        r'Report the (orthogonal projection|axial reflection \(a 180-degree rotation about the line\)) '
        r'of the (?:centre|reference point) of (.+?) using the infinite 3D line through the '
        r'(?:centres|reference points) of (.+?) and (.+?), in frame (.+?), in metres\.', source)
    if match:
        operation, target, a, b, frame = match.groups()
        operation_zh = '正交投影点' if operation == 'orthogonal projection' else '轴对称点（绕该直线旋转 180° 得到的点）'
        result = (f'以{label(a)}和{label(b)}的{point}确定一条三维无限直线，'
                  f'求{label(target)}的{point}相对于该直线的{operation_zh}。'
                  f'请在 {frame} 坐标系下报告坐标，单位为米。')
        if operation == 'orthogonal projection':
            result = (f'把{label(a)}和{label(b)}的{point}连成一条直线，并向两端延长。'
                      f'在这条线上，找到离{label(target)}的{point}最近的位置（正交投影点）。'
                      f'这是一条三维无限直线，答案可以在两个端点之外。'
                      f'请用 {frame} 坐标系给出该位置的坐标，单位为米。')
    else:
        match = re.fullmatch(
            r'Select the one object among (.+?) whose (?:centre|reference point) moved more than '
            r'([0-9.eE+-]+) metres since the end of phase (.+?), in frame (.+?)\.', source)
        if not match:
            return None
        objects, threshold, phase, frame = match.groups()
        names = '、'.join(label(item) for item in objects.split(', '))
        result = (f'在{names}中，选出从 {phase} 阶段结束至今，{point}位移超过 {threshold} 米的唯一物体。'
                  f'位移在 {frame} 坐标系下计算。')
    if origin:
        result += '每个物体的参考点均为其配置中定义的刚体原点。'
    return result
