"""Check nonempty GAC supervision through the actual refinement training graph."""
from types import MethodType
import torch
from adadepth.trainer import Trainer

torch.set_num_threads(2)
trainer=Trainer(dict(height=64,width=128,batch_size=2,device='cpu',backbone='monodepth2',
                     gac={'sampling':'all','window_size':9}))
trainer.set_stage(15)

def fixed_regions(self, outputs, *args):
    dynamic=torch.zeros((2,1,64,128),device=self.device)
    uncertain=torch.zeros_like(dynamic);ground=torch.zeros_like(dynamic)
    ground[:,:,40:]=1
    uncertain[:,:,37:40,16:24]=1
    dynamic[:,:,37:40,70:80]=1
    outputs.update(ground_mask=ground,final_mask=dynamic,mask_label_4=dynamic,
                   mask_label_embedding=uncertain,mask_wosky=dynamic+uncertain)
    return outputs

trainer.process_mask_labels=MethodType(fixed_regions,trainer)
batch={}
for frame in [0,-1,1]:
    batch['color',frame,0]=torch.rand(2,3,64,128)
    batch['color_aug',frame,0]=batch['color',frame,0].clone()
K=torch.eye(4).repeat(2,1,1)
K[:,0,0]=100;K[:,1,1]=100;K[:,0,2]=64;K[:,1,2]=32
batch['K',0]=K;batch['inv_K',0]=K.inverse()
outputs,losses=trainer.process_batch(batch)
assert losses['gac/source_anchors']>0
assert losses['gac/dynamic_anchors']>0
assert not outputs['gac_reference'].requires_grad
params=list(trainer.models['depth_ref'].parameters())
gradients=torch.autograd.grad(losses['gac/loss'],params,retain_graph=True,allow_unused=True)
assert any(g is not None and torch.count_nonzero(g)>0 for g in gradients)
assert all(g is None or torch.isfinite(g).all() for g in gradients)
losses['loss'].backward()
trainer.optimizer.step()
print({k:float(v.detach()) for k,v in losses.items() if k.startswith('gac/')})
print('GAC refinement integration: nonempty references, detached targets, finite nonzero depth gradients, optimizer step OK')
