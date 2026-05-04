from nemo.collections.asr import models as nemo_asr
import project_config as pc

# Restore the model from a .nemo file.
model = nemo_asr.EncDecRNNTModel.restore_from(pc.get_tokenizer_extract_model_path())

# Tokenizer is an attribute of the restored model.
tokenizer = model.tokenizer

# Quick sanity check.
print(tokenizer.text_to_tokens("Hello world!"))
print("Liczba tokenow:", len(tokenizer.vocab))

# Optionally save the tokenizer to disk.
tokenizer_dir = pc.get_tokenizer_dir()
model.save_tokenizers(tokenizer_dir)
