import utils as ut
import json
import logging
import os
from typing import List, Optional
import tokenizers
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import create_spt_model
from nemo.utils.data_utils import DataStoreObject

lyrics_to_train = ut.get_text()


class Args:
    def __init__(self,
                 data_root: str = "./model",
                 manifest: Optional[str] = None,
                 data_file: Optional[str] = None,
                 vocab_size: int = 1024,
                 tokenizer: str = "spe",  # choices=["spe", "wpe"]
                 spe_type: str = "bpe",  # choices=['bpe', 'unigram', 'char', 'word']
                 spe_character_coverage: float = 1.0,
                 spe_sample_size: int = -1,
                 spe_train_extremely_large_corpus: bool = False,
                 spe_max_sentencepiece_length: int = -1,
                 spe_split_by_unicode_script: bool = True,
                 spe_bos: bool = False,
                 spe_eos: bool = False,
                 spe_pad: bool = False,
                 spe_control_symbols: Optional[List[str]] = None,
                 spe_user_defined_symbols: Optional[List[str]] = None,
                 spe_byte_fallback: bool = False,
                 spe_split_digits: bool = False,
                 lower_case: bool = True,
                 log: bool = False):
        """
        Klasa args kompatybilna z process_asr_text_tokenizer.py

        Args:
            data_root: Katalog wyjściowy dla tokenizera
            manifest: Ścieżka do plików manifest (oddzielone przecinkami)
            data_file: Plik z danymi tekstowymi
            vocab_size: Rozmiar słownika
            tokenizer: Typ tokenizera ("spe" lub "wpe")
            spe_type: Typ modelu SentencePiece
            spe_character_coverage: Pokrycie znaków dla SentencePiece
            spe_sample_size: Rozmiar próbki (-1 dla całego zbioru)
            spe_train_extremely_large_corpus: Czy trenować na bardzo dużym korpusie
            spe_max_sentencepiece_length: Maksymalna długość tokenu
            spe_split_by_unicode_script: Czy dzielić według skryptu Unicode
            spe_bos: Czy dodać token <s>
            spe_eos: Czy dodać token </s>
            spe_pad: Czy dodać token <pad>
            spe_control_symbols: Symbole kontrolne
            spe_user_defined_symbols: Symbole zdefiniowane przez użytkownika
            spe_byte_fallback: Byte fallback dla nieznanych znaków
            spe_split_digits: Czy dzielić cyfry
            lower_case: Czy konwertować na małe litery
            log: Czy włączyć logowanie
        """
        self.data_root = data_root
        self.manifest = manifest
        self.data_file = data_file
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.spe_type = spe_type
        self.spe_character_coverage = spe_character_coverage
        self.spe_sample_size = spe_sample_size
        self.spe_train_extremely_large_corpus = spe_train_extremely_large_corpus
        self.spe_max_sentencepiece_length = spe_max_sentencepiece_length
        self.spe_split_by_unicode_script = spe_split_by_unicode_script
        self.spe_bos = spe_bos
        self.spe_eos = spe_eos
        self.spe_pad = spe_pad
        self.spe_control_symbols = spe_control_symbols
        self.spe_user_defined_symbols = spe_user_defined_symbols
        self.spe_byte_fallback = spe_byte_fallback
        self.spe_split_digits = spe_split_digits
        self.lower_case = lower_case
        self.log = log


args = Args(data_root="./model",manifest="./data/test_data_1.jsonl,./data/train_data_8.jsonl,./data/valid_data_1.jsonl",vocab_size=2048)
def __build_document_from_manifests(
    data_root: str, manifests: str,
):
    if ',' in manifests:
        manifests = manifests.split(',')
    else:
        manifests = [manifests]

    document_dir = os.path.join(data_root, 'text_corpus')
    if not os.path.exists(document_dir):
        os.makedirs(document_dir)

    document_path = os.path.join(document_dir, 'document.txt')

    if os.path.exists(document_path):
        logging.info('Corpus already exists at path : %s', document_path)
        return document_path

    num_lines = 0
    with open(document_path, 'w') as out_writer:
        for manifest in manifests:
            with open(DataStoreObject(manifest).get(), 'r') as in_reader:
                for line in in_reader:
                    item = json.loads(line)
                    text = item['text']

                    out_writer.write(text + '\n')
                    out_writer.flush()

                    num_lines += 1

            logging.info(f"Finished extracting manifest : {manifest}")

        logging.info("Finished extracting all manifests ! Number of sentences : {}".format(num_lines))
    return document_path


def __process_data(
    text_path: str,
    dst_folder: str,
    vocab_size: int,
    tokenizer_type: str,
    spe_type: str,
    spe_character_coverage: float,
    spe_train_extremely_large_corpus: bool,
    spe_sample_size: int,
    spe_max_sentencepiece_length: int,
    spe_split_by_unicode_script: bool,
    spe_bos: bool,
    spe_eos: bool,
    spe_pad: bool,
    spe_control_symbols: Optional[List[str]],
    spe_user_defined_symbols: Optional[List[str]],
    spe_byte_fallback: bool,
    spe_split_digits: bool,
    lower_case: bool,
):
    """
    Converts flac to wav and build manifests's json
    Args:
        text_path: source with text lines
        dst_folder: where wav files will be stored
        vocab_size: vocabular size used in encoding the text
        tokenizer_type: type of tokenization to perform - wpe or spe
        spe_type: type of tokenization model used for spe.
        spe_character_coverage: float value between 0 and 1 (as a percentage). For languages with a vast charset,
            can be < 1.0, but for all other languages, it should be set as 1.0
        spe_sample_size: int, default of -1. If positive integer is used, samples the dataset
            by given sample size.
        spe_train_extremely_large_corpus: bool. If dataset is too large, and user has sufficient RAM,
            this flag can be set to try to trained the tokenizer. Will silently fail if it runs out of RAM.
        spe_max_sentencepiece_length: Limits the maximum length of the SentencePiece subword that can be constructed.
            By default, no limit is placed.
        spe_bos: Bool flag, whether to add <s> to SentencePiece tokenizer vocabulary.
        spe_eos: Bool flag, whether to add </s> to SentencePiece tokenizer vocabulary.
        spe_pad: Bool flag, whether to add <pad> to SentencePiece tokenizer vocabulary.
        spe_control_symbols: control symbols to add to tokenizer, as defined by sentencepiece.
            These tokens get removed at decode time and are not encoded from the text - can only be added to the input programatically.
        spe_user_defined_symbols: user symbols to add to tokenizer, as defined by sentencepiece.
            These tokens remain in the decoded text and are encoded automatically when present in the input text.
        spe_byte_fallback: If <unk>, fallback to a byte sequence of the character.
        spe_split_digits: If true, digits are split into individual tokens.
        lower_case: whether to tokenize with lower case character set only (for english)

    Returns:
    """
    if tokenizer_type == 'spe':

        # Prepare directory of tokenizer
        if spe_max_sentencepiece_length > 0:
            tokenizer_dir = os.path.join(dst_folder, 'tokenizer_{}_{}_v{}_max_{}').format(
                tokenizer_type, spe_type, vocab_size, spe_max_sentencepiece_length
            )
        else:
            tokenizer_dir = os.path.join(dst_folder, 'tokenizer_{}_{}_v{}').format(
                tokenizer_type, spe_type, vocab_size
            )

        if spe_pad:
            tokenizer_dir = f'{tokenizer_dir}_pad'
        if spe_bos:
            tokenizer_dir = f'{tokenizer_dir}_bos'
        if spe_eos:
            tokenizer_dir = f'{tokenizer_dir}_eos'

        if not os.path.exists(tokenizer_dir):
            os.makedirs(tokenizer_dir)

        if os.path.exists(os.path.join(tokenizer_dir, 'tokenizer.model')):
            logging.warning("Model file already exists, overriding old model file !")
            os.remove(os.path.join(tokenizer_dir, 'tokenizer.model'))

        # Build tokenizer
        tokenizer_path, vocab_path = create_spt_model(
            data_file=text_path,
            vocab_size=vocab_size,
            sample_size=spe_sample_size,
            do_lower_case=lower_case,
            output_dir=tokenizer_dir,
            tokenizer_type=spe_type,
            character_coverage=spe_character_coverage,
            train_extremely_large_corpus=spe_train_extremely_large_corpus,
            max_sentencepiece_length=spe_max_sentencepiece_length,
            split_by_unicode_script=spe_split_by_unicode_script,
            bos=spe_bos,
            eos=spe_eos,
            pad=spe_pad,
            control_symbols=spe_control_symbols,
            user_defined_symbols=spe_user_defined_symbols,
            byte_fallback=spe_byte_fallback,
            split_digits=spe_split_digits,
        )

    else:
        tokenizer_dir = os.path.join(dst_folder, 'tokenizer_{}_v{}').format(tokenizer_type, vocab_size)

        if not os.path.exists(tokenizer_dir):
            os.makedirs(tokenizer_dir)

        tokenizer = tokenizers.BertWordPieceTokenizer(lowercase=lower_case)

        tokenizer.train(text_path, vocab_size=vocab_size)
        tokenizer.save_model(tokenizer_dir)

    return tokenizer_dir


def main():
    data_root = args.data_root
    manifests = args.manifest
    data_file = args.data_file
    vocab_size = args.vocab_size
    tokenizer = args.tokenizer
    spe_type = args.spe_type
    spe_character_coverage = args.spe_character_coverage
    spe_sample_size = args.spe_sample_size
    spe_train_extremely_large_corpus = args.spe_train_extremely_large_corpus
    spe_max_sentencepiece_length = args.spe_max_sentencepiece_length
    spe_split_by_unicode_script = args.spe_split_by_unicode_script
    spe_bos, spe_eos, spe_pad = args.spe_bos, args.spe_eos, args.spe_pad
    spe_control_symbols = args.spe_control_symbols
    spe_user_defined_symbols = args.spe_user_defined_symbols
    spe_byte_fallback = args.spe_byte_fallback
    spe_split_digits = args.spe_split_digits
    lower_case = args.lower_case

    if not os.path.exists(data_root):
        os.makedirs(data_root)

    if args.log:
        logging.basicConfig(level=logging.INFO)

    if manifests:
        text_corpus_path = __build_document_from_manifests(data_root, manifests)
    else:
        text_corpus_path = data_file
    tokenizer_path = __process_data(
        text_corpus_path,
        data_root,
        vocab_size,
        tokenizer,
        spe_type,
        lower_case=lower_case,
        spe_character_coverage=spe_character_coverage,
        spe_sample_size=spe_sample_size,
        spe_train_extremely_large_corpus=spe_train_extremely_large_corpus,
        spe_max_sentencepiece_length=spe_max_sentencepiece_length,
        spe_split_by_unicode_script=spe_split_by_unicode_script,
        spe_bos=spe_bos,
        spe_eos=spe_eos,
        spe_pad=spe_pad,
        spe_control_symbols=spe_control_symbols,
        spe_user_defined_symbols=spe_user_defined_symbols,
        spe_byte_fallback=spe_byte_fallback,
        spe_split_digits=spe_split_digits,
    )

    print("Serialized tokenizer at location :", tokenizer_path)
    logging.info('Done!')


if __name__ == "__main__":
    main()