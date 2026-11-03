import unittest
import torch
from adadepth.ground_adherent import GroundAdherentLoss


def fixture(batch=1, dtype=torch.float64):
    depth=torch.full((batch,1,12,14),10.,dtype=dtype)
    ground=torch.zeros_like(depth,dtype=torch.bool);ground[:,:,7:]=True
    uncertain=torch.zeros_like(ground);uncertain[:,:,5:7,2:4]=True
    dynamic=torch.zeros_like(ground);dynamic[:,:,5:7,9:11]=True
    depth[uncertain]=12.;depth[dynamic]=15.
    return depth,torch.zeros((batch,3,12,14),dtype=dtype),ground,uncertain,dynamic


class GACTests(unittest.TestCase):
    def test_exact_offset_reference_and_gradient(self):
        depth,image,g,u,d=fixture();depth.requires_grad_()
        result=GroundAdherentLoss({'smoothness_weight':0, 'propagation_radius':0})(depth,image,g,u,d)
        self.assertEqual(float(result['delta'][0]),2.)
        self.assertEqual(int(result['dynamic_count'][0]),4)
        torch.testing.assert_close(result['reference'][d],torch.full((4,),12.,dtype=depth.dtype))
        self.assertAlmostEqual(float(result['data_loss']),3.*4/(12*14))
        result['loss'].backward()
        torch.testing.assert_close(depth.grad[d],torch.full((4,),1/(12*14),dtype=depth.dtype))
        self.assertEqual(float(depth.grad[~d].abs().sum()),0.)
        self.assertFalse(result['reference'].requires_grad)

    def test_images_have_independent_offsets(self):
        depth,image,g,u,d=fixture(2)
        depth[1][u[1]]=9.
        result=GroundAdherentLoss()(depth,image,g,u,d)
        torch.testing.assert_close(result['delta'],torch.tensor([2.,-1.],dtype=depth.dtype))
        self.assertTrue((result['reference'][1][d[1]]==9.).all())

    def test_missing_ground_or_source_has_no_fabricated_target(self):
        for remove in ['ground','uncertain']:
            depth,image,g,u,d=fixture();depth.requires_grad_()
            if remove=='ground':g.zero_()
            else:u.zero_()
            result=GroundAdherentLoss()(depth,image,g,u,d)
            self.assertEqual(int(result['dynamic_count'].sum()),0)
            self.assertEqual(float(result['data_loss']),0.)
            result['loss'].backward()
            self.assertTrue(torch.isfinite(depth.grad).all())

    def test_chunking_is_exact(self):
        args=fixture(2)
        a=GroundAdherentLoss({'chunk_size':1, 'sampling':'all'})(*args)
        b=GroundAdherentLoss({'chunk_size':10000, 'sampling':'all'})(*args)
        for key in ['reference','ground_median','delta','loss']:
            torch.testing.assert_close(a[key],b[key])

    def test_contour_normal_rejects_vertical_ground_boundary(self):
        depth,image,g,u,d=fixture()
        g.zero_();g[:,:,:,7:]=True
        u.zero_();u[:,:,2:4,5:7]=True
        d.zero_();d[:,:,8:10,5:7]=True
        result=GroundAdherentLoss()(depth,image,g,u,d)
        self.assertEqual(int(result['source_count'].sum()),0)
        self.assertEqual(int(result['dynamic_count'].sum()),0)

    def test_no_image_border_contact(self):
        depth,image,g,u,d=fixture();g[:]=True;u.zero_();d.zero_()
        result=GroundAdherentLoss()(depth,image,g,u,d)
        self.assertEqual(int(result['source_count'].sum()),0)
        self.assertEqual(float(result['loss']),0.)

    def test_shapes_overlap_and_bad_depth_rejected(self):
        loss=GroundAdherentLoss()
        args=list(fixture())
        with self.assertRaises(ValueError):loss(args[0][:,0],*args[1:])
        args=list(fixture());args[2]|=args[3]
        with self.assertRaises(ValueError):loss(*args)
        args=list(fixture());args[0][0,0,0,0]=float('nan')
        with self.assertRaises(ValueError):loss(*args)

    def test_smoothness_ignores_object_boundary(self):
        depth,image,g,u,d=fixture();u.zero_()
        result=GroundAdherentLoss()(depth,image,g,u,d)
        self.assertEqual(float(result['smoothness_loss']),0.)
        depth[:,:,5,9:11]=14.
        result=GroundAdherentLoss()(depth,image,g,u,d)
        self.assertGreater(float(result['smoothness_loss']),0.)

    def test_small_image_larger_window_is_safe(self):
        depth=torch.ones((1,1,2,2),requires_grad=True)
        mask=torch.zeros_like(depth,dtype=torch.bool)
        result=GroundAdherentLoss()(depth,torch.zeros(1,3,2,2),mask,mask,mask)
        result['loss'].backward()
        self.assertEqual(float(result['loss']),0.)
        self.assertTrue(torch.isfinite(depth.grad).all())


class GACPluginTests(unittest.TestCase):
    def test_seeded_random_propagation_is_reproducible(self):
        args=fixture()
        loss=GroundAdherentLoss({'propagation_radius':3,'samples_per_anchor':2})
        a=loss(*args,generator=torch.Generator().manual_seed(23))
        b=loss(*args,generator=torch.Generator().manual_seed(23))
        torch.testing.assert_close(a['reference'],b['reference'])
        self.assertTrue((a['dynamic_anchors'] <= args[-1]).all())

    def test_propagation_never_crosses_disconnected_objects(self):
        depth,image,g,u,d=fixture()
        d[:,:,3:5,12:14]=True
        result=GroundAdherentLoss({'propagation_radius':5,'sampling':'all'})(depth,image,g,u,d)
        self.assertFalse(result['dynamic_anchors'][:,:,3:5,12:14].any())
        self.assertTrue(result['contact_anchors'].any())

    def test_custom_contact_and_offset_plugins(self):
        args=fixture()
        loss=GroundAdherentLoss({'propagation_radius':0},
                               contact_provider=lambda ground: torch.ones_like(ground),
                               offset_reducer=lambda values: values.mean()+1)
        result=loss(*args)
        self.assertEqual(float(result['delta'][0]),3.)
        self.assertTrue((result['reference'][args[-1]]==13.).all())

    def test_invalid_configuration(self):
        for config in [{'window_size':2},{'sampling':'bad'},{'chunk_size':0},{'propagation_radius':-1}]:
            with self.assertRaises(ValueError):GroundAdherentLoss(config)

if __name__=='__main__':unittest.main()
