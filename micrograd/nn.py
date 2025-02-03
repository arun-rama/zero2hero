import random
from typing import Sequence
from micrograd.engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
    
    def parameters(self) -> Sequence[Value]:
        return []


class Neuron(Module):
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x) -> Value:
        # w * x + b
        act = sum(wi*xi for wi,xi in zip(self.w, x)) + self.b
        out = act.tanh()
        return out
    
    def parameters(self) -> Sequence[Value]:
        return self.w + [self.b]
    
    def __repr__(self):
        return f"TanhNeuron({len(self.w)})"
    

class Layer(Module):
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x) -> Sequence[Value]:
        outs = [it(x) for it in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self) -> Sequence[Value]:
        return [p for n in self.neurons for p in n.parameters()]
    
    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
    

class MLP(Module):
    def __init__(self, nin, nouts):
        # UNDO
        # random.seed(0)
        sizes = [nin] + nouts
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(nouts))]

    def __call__(self, x) -> Sequence[Value]:
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self) -> Sequence[Value]:
        return [p for layer in self.layers for p in layer.parameters()]
    
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"