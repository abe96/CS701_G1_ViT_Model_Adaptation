import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
from models.peft_vit_improved import create_model 
from tqdm import tqdm


class InferenceDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        self.transform = transform
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_name

def main():
    # Config - change paths as needed
    data_dir = '/common/home/projectgrps/CS701/CS701G1/cs701-course-data'
    val_img_dir = os.path.join(data_dir, 'val')  # folder containing validation images
    checkpoint_path = '/common/home/projectgrps/CS701/CS701G1/transfer/src/checkpoints/best_model.pth'
    batch_size = 128  # Keep high. Lower to 512, 384, or 256 if you get OOM.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 10000  # adjust if different


    transform = transforms.Compose([
        transforms.Resize(256),         # 1. Resize shortest edge to 256
        transforms.CenterCrop(256),     # 2. Force a 256x256 square
        transforms.ToTensor(),          # 3. Convert to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    # ---

    # Dataset and loader
    val_dataset = InferenceDataset(val_img_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    # Load model
    print("Loading model checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model_config = checkpoint.get('config', {})
    model = create_model(
        num_classes=num_classes,
        peft_method=model_config.get('peft_method', 'lora'),
        rank=model_config.get('lora_rank', 16),
        alpha=model_config.get('lora_alpha', 32),
        adapter_dim=model_config.get('adapter_dim', 48),
        use_multi_scale=model_config.get('use_multi_scale', True)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("Model loaded successfully.")


    # Predict and save results
    output_lines = []
    
    # --- Start timer ---
    start_time = time.time()
    
    with torch.no_grad():
        for images, img_names in tqdm(val_loader, desc="Inference Progress"):
            images = images.to(device)
            
          
            outputs = model(images, use_tta=True)
            
            preds = torch.argmax(outputs, dim=1).cpu().tolist()
            for img_name, pred in zip(img_names, preds):
                output_lines.append(f'val/{img_name} {pred}')

    # --- Calculate and print total inference time ---
    end_time = time.time()
    inference_time = end_time - start_time

    # Save prediction file (e.g. for submission)
    out_file = 'val_team_01.txt'
    with open(out_file, 'w') as f:
        for line in output_lines:
            f.write(line + '\n')

    print(f'Saved {len(output_lines)} predictions to {out_file}')
    print(f'Total inference time: {inference_time:.2f} seconds')

if __name__ == '__main__':
    main()