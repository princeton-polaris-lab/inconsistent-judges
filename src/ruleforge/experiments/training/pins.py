from types import MappingProxyType
from typing import Mapping

#: Protected file name to the sha256 of its bytes; the training scenarios' marker must have been checked against
#: each unless the build declares otherwise.
PINNED_PROTECTED: Mapping[str, str] = MappingProxyType({
    "frozen_500": "fa9bdc5eb6b9bac8e7b42a27f4ab9f6b60bd5a0e4ac4ae3a63190eb00e487411",
    "frozen_500_fig5_grpo_disjoint": "d677125656489ce9251162f36c7c7f43716d1ba80a36fd71cf06886217885f7b",
    "wildchat_stress_weird_200x3": "77c6cf97a1c3211ed3a389c12ee9b4dccb010afd36ee9e35595d5676e44c1d41",
})

__all__ = ["PINNED_PROTECTED"]
