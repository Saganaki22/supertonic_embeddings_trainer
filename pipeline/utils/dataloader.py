import torch
from torch.utils.data import TensorDataset, DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_train_dataloader(tts, texts, batch_size=1):
    sorted_texts = sorted(
        texts,
        reverse=True,
        key=lambda x: len(tts.text_processor._preprocess_text(text=x[1], lang=x[0])),
    )
    text_languages = [t[0] for t in sorted_texts]
    train_texts = [t[1] for t in sorted_texts]

    ids_np, mask_np = tts.text_processor(train_texts, text_languages)
    input_ids = torch.tensor(ids_np, dtype=torch.long).to(DEVICE)
    attention_mask = torch.tensor(mask_np, dtype=torch.long).to(DEVICE)

    train_ds = TensorDataset(input_ids, attention_mask)
    return DataLoader(train_ds, batch_size=batch_size, shuffle=False)
