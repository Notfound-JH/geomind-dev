import unittest
from geomind.episodes import ContractError
from geomind.episodes.physical_contract import execution_profile, validate_execution, physics_profile

class MidpointPlacementContract(unittest.TestCase):
    def episode(self):
        cfg=execution_profile();cfg.update(action='PickPlaceMidpoint',moving_object='green')
        return {'task_id':'InferMidpoint','parameters':{'endpoints':['red','blue']},'physical_execution':cfg}

    def test_placement_contract_accepts_distinct_moving_object(self):
        validate_execution(self.episode(),{'robot':'franka','physics':physics_profile()})

    def test_cannot_move_reference_or_relabel_projection_as_midpoint_placement(self):
        for field,value in [('moving_object','red'),('task_id','ProjectOntoLine')]:
            ep=self.episode()
            if field=='task_id': ep[field]=value
            else: ep['physical_execution'][field]=value
            with self.subTest(field=field),self.assertRaises(ContractError):
                validate_execution(ep,{'robot':'franka','physics':physics_profile()})
