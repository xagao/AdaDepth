"""CPU-only synthetic forward/backward checks. Does not validate paper accuracy."""
import torch
from adadepth.trainer import Trainer

torch.set_num_threads(2)
config=dict(height=64,width=128,batch_size=2,device='cpu',backbone='monodepth2',model='lite-mono-8m')
trainer=Trainer(config)
for epoch in [0,10,15]:
    torch.manual_seed(43)
    batch={}
    for frame in [0,-1,1]:
        batch['color',frame,0]=torch.rand(2,3,64,128)
        batch['color_aug',frame,0]=batch['color',frame,0].clone()
    K=torch.eye(4).repeat(2,1,1)
    K[:,0,0]=100;K[:,1,1]=100;K[:,0,2]=64;K[:,1,2]=32
    batch['K',0]=K;batch['inv_K',0]=K.inverse()
    stage=trainer.set_stage(epoch)
    trainer.optimizer.zero_grad(set_to_none=True)
    outputs,losses=trainer.process_batch(batch)
    loss=losses['loss']
    assert torch.isfinite(loss),loss
    loss.backward()
    grads=[p.grad for m in trainer.models.values() for p in m.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    frozen=[p for m in trainer.models.values() for p in m.parameters() if not p.requires_grad]
    assert all(p.grad is None for p in frozen)
    trainer.optimizer.step()
    print(stage,'loss=',float(loss.detach()),'finite_grad_tensors=',len(grads),flush=True)
