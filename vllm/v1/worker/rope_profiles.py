# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RequestStaticYarnConfig:
    factors: tuple[float, ...]
    factor_offsets: dict[float, int]
    original_max_position: int

    @classmethod
    def from_model_config(cls, model_config: Any) -> "RequestStaticYarnConfig | None":
        rope_parameters = (
            getattr(model_config.hf_text_config, "rope_parameters", None) or {}
        )
        configured_factors = rope_parameters.get("request_static_factors")
        factor_env = os.getenv("VLLM_REQUEST_STATIC_YARN_FACTORS")
        if factor_env:
            factors = tuple(float(factor) for factor in factor_env.split(","))
            if (
                configured_factors is not None
                and tuple(float(factor) for factor in configured_factors) != factors
            ):
                raise ValueError(
                    "Request-static YaRN environment and model factors differ"
                )
        elif configured_factors is not None:
            factors = tuple(float(factor) for factor in configured_factors)
        else:
            return None

        if not factors or factors != tuple(sorted(set(factors))):
            raise ValueError(
                "Request-static YaRN factors must be non-empty, unique, and sorted"
            )
        if any(factor < 1.0 for factor in factors):
            raise ValueError("Request-static YaRN factors must be at least 1")

        original_max_position = int(rope_parameters["original_max_position_embeddings"])
        offsets: dict[float, int] = {}
        offset = 0
        for factor in factors:
            offsets[factor] = offset
            offset += int(4 * original_max_position * factor)
        return cls(factors, offsets, original_max_position)

    def select_factor(self, required_tokens: int) -> float:
        return select_request_static_yarn_factor(
            required_tokens,
            self.original_max_position,
            self.factors,
        )

    def position_offset(self, required_tokens: int) -> int:
        return self.factor_offsets[self.select_factor(required_tokens)]

    def validate_serving_config(self, vllm_config: Any) -> None:
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError(
                "Request-static YaRN requires prefix caching to be disabled "
                "until the RoPE profile is part of each block hash"
            )
        if vllm_config.kv_transfer_config is not None:
            raise ValueError(
                "Request-static YaRN requires KV transfer/offload to be disabled "
                "until the RoPE profile is part of the external cache identity"
            )


def select_request_static_yarn_factor(
    required_tokens: int,
    original_max_position: int,
    factors: tuple[float, ...],
) -> float:
    """Choose the smallest configured profile that covers a request budget."""
    for factor in factors:
        if required_tokens <= original_max_position * factor:
            return factor
    raise ValueError(
        f"Request needs {required_tokens} tokens, but the largest request-static "
        f"YaRN profile covers {original_max_position * factors[-1]:g}"
    )
