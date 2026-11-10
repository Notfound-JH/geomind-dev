import unittest
from geomind.tools.instruction_zh import translate_instruction

class TestInstructionZh(unittest.TestCase):
    def test_projection_uses_recorded_names(self):
        s='Report the orthogonal projection of the centre of the green point using the infinite 3D line through the centres of the red endpoint and the blue endpoint, in frame table, in metres.'
        zh=translate_instruction(s)
        for word in ['绿色参考物体', '红色端点物体', '蓝色端点物体', '三维无限直线', '正交投影点', '单位为米']:
            self.assertIn(word,zh)
        self.assertNotIn('刚体原点',zh)

    def test_reflection_and_origin_are_explicit(self):
        s="Report the axial reflection (a 180-degree rotation about the line) of the reference point of the yellow mustard bottle using the infinite 3D line through the reference points of the coffee can and the meat tin, in frame table, in metres. Each object's reference point is its configured rigid-body origin."
        zh=translate_instruction(s)
        for word in ['180°','黄色芥末瓶','咖啡罐','肉罐','刚体原点']: self.assertIn(word,zh)

    def test_memory_threshold_and_phase_preserved(self):
        s='Select the one object among the red block, the blue block, the green block whose centre moved more than 0.03 metres since the end of phase observe, in frame table.'
        zh=translate_instruction(s)
        for word in ['超过 0.03 米','observe 阶段结束','唯一物体','红色方块']: self.assertIn(word,zh)

    def test_unknown_template_is_not_guessed(self):
        self.assertIsNone(translate_instruction('Pick up a cup.'))
