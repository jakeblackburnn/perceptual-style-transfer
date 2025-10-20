class ActivationExtractor:
    def __init__(self, vgg_model):
        self.activations = {}
        self.hooks = []
        self._register_hooks(vgg_model)

    def _register_hooks(self, model):
        def hook_fn(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook

        for name, module in model.vgg._modules.items():
            hook = module.register_forward_hook(hook_fn(name))
            self.hooks.append(hook)
