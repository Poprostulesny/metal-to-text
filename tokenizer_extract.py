from nemo.collections.asr import models as nemo_asr

# przywrócenie modelu z pliku .nemo
model = nemo_asr.EncDecRNNTModel.restore_from("./model/parakeet-rnnt-1.1b.nemo")

# tokenizator jest atrybutem modelu
tokenizer = model.tokenizer

# sprawdzenie działania
print(tokenizer.text_to_tokens("Hello world!"))
print("Liczba tokenów:", len(tokenizer.vocab))

# opcjonalnie: zapis tokenizatora na dysk
tokenizer_dir = "model/model_tokenizer"
model.save_tokenizers(tokenizer_dir)
