import os
import json
import torch
from torch_geometric.data import Data
from transformers import BertTokenizer, BertModel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def preprocess_object(obj, caption_text):
    vertices = torch.tensor(obj['v'], dtype=torch.float, device=device)
    edges = torch.tensor(obj['e'], dtype=torch.long, device=device).t()  # Transpose to shape [2, num_edges]

    print("Caption:", caption_text)

    inputs = tokenizer(
        caption_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=32
    )
    input_ids = inputs["input_ids"].to(device=device, dtype=torch.long)
    
    with torch.no_grad():
        outputs = bert_model(input_ids)
    # Pool the embeddings: take the mean over the sequence dimension
    text_label_embedding = outputs.last_hidden_state.mean(dim=1)  # [batch_size, embed_dim]

    # Create a PyG Data object with node features, edge index, and the text embedding
    graph_data = Data(
        x=vertices,
        edge_index=edges,
        text_label=text_label_embedding
    )
    return graph_data

def load_json_and_caption(json_file_path):
    with open(json_file_path, 'r') as f:
        obj = json.load(f)

    base_name = os.path.splitext(os.path.basename(json_file_path))[0]
    folder_path = os.path.dirname(json_file_path)

    caption_text = ""
    for file_name in os.listdir(folder_path):
        if file_name.startswith(base_name) and file_name.endswith("_caption.txt"):
            caption_file_path = os.path.join(folder_path, file_name)
            with open(caption_file_path, 'r', encoding='utf-8') as cf:
                caption_text = cf.read().strip()
            break

    if not caption_text:
        print(f"No matching caption file found for {base_name}, using obj['n'] if available.")
        caption_text = obj.get('n', '')

    return preprocess_object(obj, caption_text)

def load_single_data_point(root_folder, max_nodes=5000):
 
    for model_id in os.listdir(root_folder):
        model_folder = os.path.join(root_folder, model_id)
        if os.path.isdir(model_folder):
            json_filename = f"{model_id}.json"
            json_path = os.path.join(model_folder, json_filename)
            if os.path.exists(json_path):
                data = load_json_and_caption(json_path)
                num_nodes = data.x.size(0)
                if num_nodes > max_nodes:
                    print(f"Skipping {json_path} due to excessive nodes: {num_nodes} (max allowed {max_nodes}).")
                    continue
                print(f"Loaded data from {json_path} with {num_nodes} nodes.")
                return data  # Return the first valid data point
            else:
                print(f"No JSON file found for {model_id} in {model_folder}. Skipping.")
        else:
            print(f"{model_id} is not a directory. Skipping.")
    return None
