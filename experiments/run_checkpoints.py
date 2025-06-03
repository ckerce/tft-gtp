from model.model_token_factored_alibi import FactoredTransformerModelALiBi as ModelALiBi

model, tokenizer = ModelALiBi.load_from_checkpoint('output_alibi/alibi_model.pt',device='cuda')