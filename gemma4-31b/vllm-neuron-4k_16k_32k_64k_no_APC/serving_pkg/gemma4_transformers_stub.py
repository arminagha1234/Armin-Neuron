# SPDX-License-Identifier: Apache-2.0
"""Minimal transformers stub so AutoConfig recognizes Gemma4.

transformers 4.57.x does not ship the `gemma4` / `gemma4_text` model types, so
`AutoConfig.from_pretrained("google/gemma-4-31b-it")` raises
  ValueError: ... model type `gemma4` ... Transformers does not recognize ...

vLLM calls AutoConfig before it ever reaches our registered Neuron model, so we
must teach transformers about the config types. We do NOT need full modeling
classes here — vLLM-Neuron builds the model from our custom `gemma4` package via
`from_configs(hf_config, ...)`. We only need a PretrainedConfig subclass that:
  (1) carries the nested text_config / audio_config dicts through, and
  (2) is registered under model_type "gemma4" (and "gemma4_text").

This mirrors what the beta's install_transformers_stub.sh does, but as an
importable module loaded via sitecustomize (no site-packages surgery).
"""
import logging

from transformers import AutoConfig, PretrainedConfig

logger = logging.getLogger(__name__)


class Gemma4TextConfig(PretrainedConfig):
    model_type = "gemma4_text"

    def __init__(self, **kwargs):
        # Keep all keys as attributes; our gemma4.config.Gemma4Config.from_configs
        # extracts what it needs. Pull a few commonly-accessed ones to top level.
        super().__init__(**kwargs)


class Gemma4Config(PretrainedConfig):
    model_type = "gemma4"
    # Tell transformers the sub-config field -> class mapping so nested dicts
    # become typed sub-configs rather than raw dicts.
    sub_configs = {"text_config": Gemma4TextConfig}

    def __init__(self, text_config=None, **kwargs):
        if isinstance(text_config, dict):
            text_config = Gemma4TextConfig(**text_config)
        self.text_config = text_config
        super().__init__(**kwargs)


def install() -> None:
    """Idempotently register the gemma4 config types with transformers Auto*."""
    registered = []
    for mt, cls in [("gemma4_text", Gemma4TextConfig), ("gemma4", Gemma4Config)]:
        try:
            AutoConfig.register(mt, cls, exist_ok=True)
            registered.append(mt)
        except TypeError:
            # older transformers: register() has no exist_ok kwarg
            try:
                AutoConfig.register(mt, cls)
                registered.append(mt)
            except ValueError:
                pass  # already registered
            except Exception as e:
                logger.warning("AutoConfig.register(%s) failed: %r", mt, e)
        except ValueError:
            pass  # already registered
        except Exception as e:
            logger.warning("AutoConfig.register(%s) failed: %r", mt, e)
    logger.info("[gemma4_transformers_stub] registered: %s", registered or "already present")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    install()
    from transformers import AutoConfig as AC
    print("gemma4 registered:", "gemma4" in str(AC._mapping._extra_content) if hasattr(AC, "_mapping") else "?")
