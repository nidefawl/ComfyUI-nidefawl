import importlib

node_list = [
    "nidefawl",
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for module_name in node_list:
    imported_module = importlib.import_module(
        ".{}".format(module_name), __name__
    )

    NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **imported_module.NODE_CLASS_MAPPINGS}
    if hasattr(imported_module, "NODE_DISPLAY_NAME_MAPPINGS"):
        NODE_DISPLAY_NAME_MAPPINGS = {
            **NODE_DISPLAY_NAME_MAPPINGS,
            **imported_module.NODE_DISPLAY_NAME_MAPPINGS,
        }

WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

def patch_default_execute_flush_cache():
  import execution

  def nidefawl_execute_flush_cache(self, *args, **kwargs):
    if len(args) > 2 and 'extra_pnginfo' in args[2] and 'workflow' in args[2]['extra_pnginfo']:
      workflow = args[2]['extra_pnginfo']['workflow']
      is_flush_cache = 'config' in workflow and 'flush_cache' in workflow['config'] and bool(workflow['config']['flush_cache'])
      if is_flush_cache:
        self.outputs = {}
    
    return self.default_execute(*args, **kwargs)

  execution.PromptExecutor.default_execute = execution.PromptExecutor.execute
  execution.PromptExecutor.execute = nidefawl_execute_flush_cache

def patch_torch_nn_module_load_state_dict():
  import torch.nn.modules.module
  from typing import Mapping, Any
  STATE_DICT_ASSIGN = True
  def nidefawl_load_state_dict(self, state_dict: Mapping[str, Any],
                        strict: bool = True, assign: bool = True):
    assign = STATE_DICT_ASSIGN
    return self.default_load_state_dict(state_dict, strict, assign)

  torch.nn.modules.module.Module.default_load_state_dict = torch.nn.modules.module.Module.load_state_dict
  torch.nn.modules.module.Module.load_state_dict = nidefawl_load_state_dict

patch_default_execute_flush_cache()
patch_torch_nn_module_load_state_dict()