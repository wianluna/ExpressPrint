from typing import Dict

import torch


def dump_config(config, writer) -> None:
    def _walk(section: dict, param_name: str) -> None:
        for key, value in section.items():
            sub_param_name = f"{param_name}.{key}" if param_name != "" else key

            if isinstance(value, dict):
                _walk(value, sub_param_name)
            elif isinstance(value, list):
                _walk(dict(enumerate(value)), sub_param_name)
            else:
                if torch.is_tensor(value):
                    params[sub_param_name] = str(value.item())
                else:
                    params[sub_param_name] = str(value)

    params: Dict[str, str] = {}
    _walk(config, param_name="")
    for key, value in params.items():
        writer.add_text(key, value)
