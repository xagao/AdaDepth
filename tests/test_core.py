import unittest
import tempfile
from pathlib import Path
import torch
from adadepth.paper_losses import motion_robust, masked_mean_over_image, residual_sparsity, refinement_objective
from adadepth.metrics import depth_metrics
from adadepth.models import load_component
from layers import GroundPlane

class CoreTests(unittest.TestCase):
    def test_masked_batch_has_no_cross_sample_broadcast(self):
        raw = torch.tensor([1., 10.]).view(2,1,1,1)
        pseudo = torch.tensor([2., 3.]).view(2,1,1,1)
        dynamic = torch.tensor([1., 0.]).view(2,1,1,1)
        self.assertAlmostEqual(float(motion_robust(raw,pseudo,dynamic)),1.5)
        with self.assertRaises(ValueError): masked_mean_over_image(raw[:,0],dynamic)

    def test_exact_depth_metrics(self):
        gt = torch.tensor([1.,2.,5.])
        result = depth_metrics(gt,gt)
        torch.testing.assert_close(result,torch.tensor([0.,0.,0.,0.,1.,1.,1.]))
        with self.assertRaises(ValueError): depth_metrics(gt[:0],gt[:0])

    def test_l1_sparsity_is_not_probability_bce(self):
        flow = torch.full((1,2,1,2),5.,requires_grad=True)
        photo = torch.tensor([0.,2.]).view(1,1,1,2)
        loss = residual_sparsity(flow,photo)
        self.assertAlmostEqual(float(loss),5.)
        loss.backward()
        self.assertTrue(torch.isfinite(flow.grad).all())

    def test_missing_paper_loss_does_not_silently_become_zero(self):
        with self.assertRaises(NotImplementedError): refinement_objective(1.,2.,None,3.)

    def test_partial_checkpoint_rejected(self):
        model = torch.nn.Linear(2,2)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'partial.pth'
            torch.save({'weight':model.weight.detach()},path)
            with self.assertRaises(RuntimeError): load_component(model,path)
            torch.save({'module.'+k:v for k,v in model.state_dict().items()},path)
            load_component(model,path)

    def test_ground_plane_batch_order(self):
        # Distinct planes reveal accidental interleaving of samples/RANSAC trials.
        plane = GroundPlane(max_it=8)
        x,z = torch.meshgrid(torch.linspace(-1,1,20),torch.linspace(1,3,20),indexing='ij')
        a=torch.stack([x,0.1*x+0.2*z+1,z]).reshape(3,-1).T
        b=torch.stack([x,-0.3*x+0.1*z+2,z]).reshape(3,-1).T
        points=torch.stack([a,b])
        params=plane.estimate_ground_plane(points)
        residual=plane.dist_from_plane(points,params)
        self.assertLess(float(residual.abs().max()),0.01)

if __name__ == '__main__': unittest.main()
