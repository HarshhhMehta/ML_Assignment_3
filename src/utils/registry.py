import os
import json
import time

REGISTRY_PATH = 'experiments/registry.json'

def log_experiment(exp_id, question, model_name, param_count, config,
                   val_metric, test_metric=None, checkpoint_path=None,
                   reused_from=None, notes=None):
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    registry = {}
    if os.path.exists(REGISTRY_PATH):
        try:
            with open(REGISTRY_PATH, 'r') as f:
                registry = json.load(f)
        except Exception:
            registry = {}
            
    record = {
        'experiment_id': exp_id,
        'question': question,
        'model_name': model_name,
        'param_count': param_count,
        'config': config,
        'val_metric': val_metric,
        'test_metric': test_metric,
        'checkpoint_path': checkpoint_path,
        'reused_from': reused_from,
        'notes': notes,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    registry[exp_id] = record
    with open(REGISTRY_PATH, 'w') as f:
        json.dump(registry, f, indent=2)
    print(f"Logged experiment {exp_id} to registry.")
    return record

def get_experiment(exp_id):
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, 'r') as f:
            registry = json.load(f)
        return registry.get(exp_id, None)
    return None
