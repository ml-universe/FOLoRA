"""配置序列化测试（断点续训依赖）。"""

from peft_cl.utils.config import CLConfig


def test_config_json_roundtrip():
    c = CLConfig(method="folora", lora_rank=8, tag="lam1.0")
    c2 = CLConfig.from_dict(c.to_dict())
    assert c2 == c
    assert c2.tag == "lam1.0" and c2.lora_rank == 8
