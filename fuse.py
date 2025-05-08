from torch.utils.data import Dataset, DataLoader, random_split
from peft import LoraConfig, get_peft_model
from datasets import load_dataset 
import torch, gc

gc.collect()
torch.cuda.empty_cache()
torch.manual_seed(2025)

dataset = load_dataset("imagefolder", data_dir="./images", split="train")

class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(images=item["image"], padding="max_length", return_tensors="pt")
        # remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["text"] = item["text"]
        return encoding

def collate_fn(batch):
    # pad the input_ids and attention_mask
    processed_batch = {}
    for key in batch[0].keys():
        if key != "text":
            processed_batch[key] = torch.stack([example[key] for example in batch])
        else:
            text_inputs = processor.tokenizer(
                [example["text"] for example in batch], padding=True, return_tensors="pt"
            )
            processed_batch["input_ids"] = text_inputs["input_ids"]
            processed_batch["attention_mask"] = text_inputs["attention_mask"]
    return processed_batch



from transformers import BlipProcessor, BlipForConditionalGeneration, BitsAndBytesConfig

quant_config = BitsAndBytesConfig(load_in_8bit=True)

model_id = "noamrot/FuseCap"
processor = BlipProcessor.from_pretrained("noamrot/FuseCap")
model = BlipForConditionalGeneration.from_pretrained("noamrot/FuseCap", device_map="cuda:0")
print(model)


config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.08,
    bias="lora_only",
    target_modules=["query", "key", "value"]
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

dataset = ImageCaptioningDataset(dataset, processor)

total_size = len(dataset)
val_size = int(total_size * 0.05)   
train_size = total_size - val_size  

# Split dataset
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=8, collate_fn=collate_fn)

print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")


optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

best_val_loss = float('inf')
for epoch in range(256):
    print(f"Epoch {epoch + 1}/{256}")
    
    # Training loop
    model.train()
    running_train_loss = 0.0
    for idx, batch in enumerate(train_dataloader):
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device, torch.float16)

        outputs = model(input_ids=input_ids,
                        pixel_values=pixel_values,
                        labels=input_ids)

        loss = outputs.loss
        running_train_loss += loss.item()

        print(f"Batch {idx + 1}/{len(train_dataloader)} - Training Loss: {loss.item()}")

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
    
    avg_train_loss = running_train_loss / len(train_dataloader)
    print(f"Average Training Loss for Epoch {epoch + 1}: {avg_train_loss}")
    
    # Validation loop
    model.eval()  
    running_val_loss = 0.0
    with torch.no_grad():  
        for batch in val_dataloader:  
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device, torch.float16)

            outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            labels=input_ids)

            val_loss = outputs.loss
            running_val_loss += val_loss.item()

    avg_val_loss = running_val_loss / len(val_dataloader)
    print(f"Validation Loss for Epoch {epoch + 1}: {avg_val_loss}")
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        save_directory = "./fuse200"
        model.save_pretrained(save_directory)
        processor.save_pretrained(save_directory)


