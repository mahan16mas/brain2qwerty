from xp_config import experiment_config
from neuraltrain.models import BaseModelConfig as ModelConfig
import torch 

if __name__=="__main__":
    K = 4
    N = 2
    C = 4 

    X = torch.zeros(K, N, C)

    unit_embeds = torch.ones((N, C))

    print(X + unit_embeds[None, ...])
    exit()

    cfg = experiment_config()
    cfg["brain_model_config"]["conv_dropout"] = 0.5
    cfg["brain_model_config"]["dropout_input"] = 0.2    

    brain_config = ModelConfig(**cfg["brain_model_config"])
    hidden_dim = brain_config.hidden

    brain_model = brain_config.build(n_in_channels=192, n_outputs=hidden_dim)

    print(brain_model)

    x = torch.randn([616, 192, 4])
    brain_model(x, None, None)

    transformer_config = ModelConfig(**cfg["transformer_config"])
    transformer_model = transformer_config.build(dim=hidden_dim)