"""Compilation and Tensor Shape Verification Test."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from models.resnet50 import ResNet50Backbone
from models.msf import MultiScaleFusion
from models.cbam import CBAM
from models.trustoct import TrustOCT

def test_pipeline():
    print("Testing Tensor Shapes Across TrustOCT Pipeline...")
    x = torch.randn(2, 3, 224, 224)

    # 1. ResNet50 dual-layer backbone
    backbone = ResNet50Backbone(pretrained=False)
    l3, l4 = backbone(x)
    print(f"  * Backbone Layer 3 Output : {list(l3.shape)}  (Expected: [2, 1024, 14, 14])")
    print(f"  * Backbone Layer 4 Output : {list(l4.shape)}  (Expected: [2, 2048, 7, 7])")
    assert l3.shape == torch.Size([2, 1024, 14, 14]), f"Layer 3 shape mismatch: {l3.shape}"
    assert l4.shape == torch.Size([2, 2048, 7, 7]), f"Layer 4 shape mismatch: {l4.shape}"

    # 2. Multi-Scale Feature Fusion (MSF)
    msf = MultiScaleFusion(in_channels_l3=1024, out_channels=2048)
    fused = msf(l3, l4)
    print(f"  * MSF Fused Feature Map    : {list(fused.shape)}  (Expected: [2, 2048, 7, 7])")
    assert fused.shape == torch.Size([2, 2048, 7, 7]), f"Fused shape mismatch: {fused.shape}"

    # 3. Dual CBAM Attention
    cbam = CBAM(gate_channels=2048)
    att = cbam(fused)
    print(f"  * CBAM Refined Output     : {list(att.shape)}  (Expected: [2, 2048, 7, 7])")
    assert att.shape == torch.Size([2, 2048, 7, 7]), f"CBAM shape mismatch: {att.shape}"

    # 4. Full TrustOCT Model
    model = TrustOCT(
        backbone_name="resnet50",
        pretrained=False,
        use_multiscale=True,
        use_cbam=True,
        head_type="softmax",
        num_classes=4
    )
    logits = model(x)
    print(f"  * TrustOCT Final Logits   : {list(logits.shape)}  (Expected: [2, 4])")
    assert logits.shape == torch.Size([2, 4]), f"Logits shape mismatch: {logits.shape}"

    print("\n[OK] ALL TENSOR SHAPES & MODULE COMPILATIONS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_pipeline()
