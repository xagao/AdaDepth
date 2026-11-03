"""Exercise the evaluation CLI on a generated KITTI-format fixture, not a benchmark."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from PIL import Image
import torch
from adadepth.models import depth_pair

torch.set_num_threads(2)
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp)
    image_dir=root/'data/sequence/image_02/data'
    image_dir.mkdir(parents=True)
    rng=np.random.default_rng(7)
    for i in range(2):
        Image.fromarray(rng.integers(0,256,(64,128,3),dtype=np.uint8)).save(image_dir/('%010d.png'%i))
    (root/'split.txt').write_text('sequence 0 l\nsequence 1 l\n')
    np.savez(root/'gt_depths.npz',data=np.full((2,64,128),10.,dtype=np.float32))
    weights=root/'weights';weights.mkdir()
    enc,dec=depth_pair('monodepth2',64,128)
    torch.save(enc.state_dict(),weights/'encoder_ref.pth')
    torch.save(dec.state_dict(),weights/'depth_ref.pth')
    output=root/'result.json'
    subprocess.run([sys.executable,'evaluate_depth.py','--dataset','kitti','--data-path',str(root/'data'),
                    '--weights',str(weights),'--backbone','monodepth2',
                    '--height','64','--width','128','--split-file',str(root/'split.txt'),
                    '--gt-path',str(root/'gt_depths.npz'),'--output',str(output),
                    '--device','cpu','--num-workers','0','--batch-size','2'],check=True)
    result=json.loads(output.read_text())
    assert result['processed']==2 and not result['partial'] and result['skipped']==0
    assert all(np.isfinite(v) for v in result['image_mean_metrics'].values())
    print('evaluation CLI: 2 generated images, strict weights, finite metrics, JSON output OK')
