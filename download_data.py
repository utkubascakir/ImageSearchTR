import os, json
from datasets import Dataset, load_dataset

# LOAD DATASET
stream = load_dataset(
    "ituperceptron/image-captioning-turkish",
    split="long_captions",
    streaming=True
)
stream = stream.shuffle(buffer_size=10_000, seed=14)
subset_iter = stream.take(10000)

def gen():
    for ex in subset_iter:
        yield ex

img_txt_ds = Dataset.from_generator(gen, features=stream.features)
print(f"Dataset is loaded, {img_txt_ds}")



# SAVE CAPTIONS AND PATHS AS JSONL (for encode.py)
def save_as_jsonl(save_path: str, dataset: Dataset):
    os.makedirs(save_path, exist_ok=True)
    jsonl_path = os.path.join(save_path, "captions.jsonl")
    
    try:
        with open(jsonl_path, "w", encoding='utf-8') as f:
            for item in dataset:
                img = item['image'].convert("RGB")
                text = item['text']
                img_id = item['image_id']
                
                img_file = f"{img_id}.jpeg"
                img.save(os.path.join(save_path, img_file))
                
                json_item = {
                    'path': img_file,
                    'caption': text
                }
                f.write(json.dumps(json_item, ensure_ascii=False) + "\n")
        print(f"All paths and captions are successfully saved to: {jsonl_path}")
        
    except Exception as e:
        print(f"Failed to save dataset: {str(e)}")
        
save_as_jsonl("demo/", img_txt_ds)