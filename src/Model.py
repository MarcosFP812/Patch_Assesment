from transformers import PreTrainedModel, LlamaConfig, AutoModel, AutoConfig
from torch import nn
import torch

class PatchALlama(PreTrainedModel):
    """
    Creates a DeBERTa model for fault localization.
    """

    config_class = LlamaConfig

    def __init__(self, model_path):
        super(PatchALlama, self).__init__(AutoConfig.from_pretrained(model_path))

        self.llama = AutoModel.from_pretrained(model_path, device_map="auto")

        self.linear = nn.Linear(self.llama.config.hidden_size, 1)

    def first_eots_pos(self, input_ids):
        r_tensor = []

        for input in input_ids:

            condicion = (input == 128001)
            r_tensor.append(torch.nonzero(condicion)[0].item())

        return r_tensor

    def get_last_tokens(self, last_hidden_state, input_ids):
        ids = self.first_eots_pos(input_ids)

        tokens = []

        for i in range(len(input_ids)):
            tokens.append(last_hidden_state[i][ids[i]])

        return torch.stack(tokens)
        
    def forward(self, input_ids, attention_mask,labels):

        output = self.llama(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output["last_hidden_state"]
        #Del last hidden state tengo que buscar el último eos que es el que tiene todo el contexto del fragmento del batch
        last_token = self.get_last_tokens(last_hidden_state, input_ids)

        output_lin = self.linear(last_token).squeeze(-1)
        # Output: [batch_size, number_of_lines, 1] -> [batch_size, number_of_lines]

        # Flatten output tensor
        output = output_lin.view(-1)

        # Labels 
        loss_fnc = nn.BCEWithLogitsLoss()
        loss = loss_fnc(output, labels.float())

        return (loss, output) if loss is not None else output