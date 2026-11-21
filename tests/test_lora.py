import pytest
import torch
from torch import nn

from mmft.lora import LoRALinear, apply_lora, lora_parameters, merge_lora


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(16, 16)
        self.up_proj = nn.Linear(16, 32, bias=False)
        self.other = nn.Linear(32, 8)

    def forward(self, x):
        return self.other(torch.relu(self.up_proj(self.q_proj(x))))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = nn.ModuleList([Block(), Block()])
        self.vision = Block()

    def forward(self, x):
        for b in self.language_model:
            x = b(x)[..., :16].repeat(1, 2)[..., :16]
        return x


def test_zero_init_is_identity():
    torch.manual_seed(0)
    base = nn.Linear(16, 24)
    x = torch.randn(5, 16)
    ref = base(x)
    assert torch.equal(LoRALinear(base, r=4, alpha=8)(x), ref)


def test_merge_matches_unmerged():
    torch.manual_seed(0)
    lora = LoRALinear(nn.Linear(16, 24), r=4, alpha=8)
    nn.init.normal_(lora.lora_B)
    x = torch.randn(5, 16)
    torch.testing.assert_close(lora.merged()(x), lora(x), rtol=1e-5, atol=1e-5)


def test_bf16_base_keeps_fp32_adapter():
    base = nn.Linear(16, 16).to(torch.bfloat16)
    lora = LoRALinear(base, r=2, alpha=4)
    assert lora.lora_A.dtype == torch.float32
    y = lora(torch.randn(3, 16, dtype=torch.bfloat16))
    assert y.dtype == torch.bfloat16
    assert lora.merged().weight.dtype == torch.bfloat16


def test_apply_scoped_and_merge_roundtrip():
    torch.manual_seed(0)
    m = Model()
    for p in m.parameters():
        p.requires_grad_(False)
    names = apply_lora(m, scope="language_model", targets=("q_proj", "up_proj"), r=4, alpha=8)
    assert sorted(names) == sorted(f"language_model.{i}.{t}" for i in range(2) for t in ("q_proj", "up_proj"))
    assert isinstance(m.vision.q_proj, nn.Linear)  # outside scope: untouched
    params = lora_parameters(m)
    assert sum(p.numel() for p in params) == 2 * (4 * (16 + 16) + 4 * (16 + 32))

    x = torch.randn(3, 16)
    m(x).sum().backward()
    assert all(p.grad is not None for p in params)
    assert all(p.grad is None for n, p in m.named_parameters() if ".lora_" not in n)

    for p in params:
        nn.init.normal_(p, std=0.1)
    before = m(x)
    assert merge_lora(m) == 4
    assert not any(isinstance(mod, LoRALinear) for mod in m.modules())
    torch.testing.assert_close(m(x), before, rtol=1e-5, atol=1e-5)


def test_no_match_raises():
    with pytest.raises(ValueError):
        apply_lora(Model(), scope="language_model", targets=("k_proj",), r=2, alpha=2)
