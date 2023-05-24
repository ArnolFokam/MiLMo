from src.transforms.operations import Compose, RandomCropPad, ToVocabID

PAD = '<pad>'
EOB = '<EOS>'
SOB = '<SOS>'
UNK = '<UNK>'

RESERVED_TOKENS=[PAD, EOB, SOB, UNK]

pipelines = lambda cfg, vocab: {
    "training": Compose([
        ToVocabID(vocab),
        RandomCropPad(
            min_crop_len=cfg.min_crop_len,
            max_seq_len=cfg.max_seq_len,
            padding_token=vocab.lookup_indices([PAD])[0]
        )
    ]),
    "generation": Compose([
        ToVocabID(vocab)
    ])
}