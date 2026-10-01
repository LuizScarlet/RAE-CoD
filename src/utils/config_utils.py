import yaml
import os
import copy

# Global symbol table
GLOBAL_SYMBOLS = {}

# YAML !load / !define tags
class SymbolRef:
    def __init__(self, name):
        self.name = name

class SymbolDefine:
    def __init__(self, name, value):
        self.name = name
        self.value = value

def load_constructor(loader, node):
    name = loader.construct_scalar(node)
    return SymbolRef(name)

def define_constructor(loader, node):
    parts = loader.construct_scalar(node).split(None, 1)
    if len(parts) != 2:
        raise ValueError("!define syntax must be: !define <name> <value>")
    name, value_str = parts
    value = yaml.safe_load(value_str)
    return SymbolDefine(name, value)

yaml.add_constructor('!load', load_constructor, Loader=yaml.SafeLoader)
yaml.add_constructor('!define', define_constructor, Loader=yaml.SafeLoader)

# Merge and path helpers
def deep_merge(base, update):
    if not isinstance(base, dict) or not isinstance(update, dict):
        return update
    
    for k, v in update.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base

def expand_dot_keys(data):
    if not isinstance(data, dict):
        return data
    
    dot_keys = [k for k in list(data.keys()) if isinstance(k, str) and '.' in k]
    for key in dot_keys:
        value = data.pop(key)
        parts = key.split('.')
        nested_update = {}
        current = nested_update
        for part in parts[:-1]:
            current[part] = {}
            current = current[part]
        current[parts[-1]] = value
        deep_merge(data, nested_update)
    return data

def load_yaml_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"YAML file not found: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    
    return process_data(data)

# Stage 1: expand structures and collect definitions
def process_data(data):
    if isinstance(data, list):
        out = []
        for item in data:
            v = process_data(item)
            if v is not None:
                out.append(v)
        return out
    
    if isinstance(data, dict):
        data = expand_dot_keys(data)

        base_path = data.pop('load', None)

        for k, v in data.items():
            data[k] = process_data(v)

        if base_path:
            base_data = load_yaml_file(base_path)
            return deep_merge(base_data, data)

        return data

    if isinstance(data, SymbolDefine):
        value = process_data(data.value)
        if data.name not in GLOBAL_SYMBOLS:
            GLOBAL_SYMBOLS[data.name] = value
        return value

    if isinstance(data, SymbolRef):
        # Resolve after all definitions have been collected.
        return data

    return data

# Stage 2: resolve symbol references
def resolve_symbols(data):
    if isinstance(data, list):
        return [resolve_symbols(v) for v in data]

    if isinstance(data, dict):
        return {k: resolve_symbols(v) for k, v in data.items()}

    if isinstance(data, SymbolRef):
        if data.name not in GLOBAL_SYMBOLS:
            raise KeyError(f"Symbol '{data.name}' is not defined")
        return GLOBAL_SYMBOLS[data.name]

    return data

# Public entry point
def load_config(data, overrides=None):
    GLOBAL_SYMBOLS.clear()
    
    result = process_data(copy.deepcopy(data))
    result = resolve_symbols(result)

    if overrides:
        overrides = expand_dot_keys(copy.deepcopy(overrides))
        result = deep_merge(result, overrides)

    return result
