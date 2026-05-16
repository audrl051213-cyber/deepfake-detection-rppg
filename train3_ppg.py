# 하이퍼 파라미터 설정할 수 있도록 수정, 데이터 증강도 넣음
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import os
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# [신규] 하이퍼파라미터 및 설정을 한 곳에서 관리하기 위한 구성 클래스
class Config:
    ROOT_DIR = '/kaggle/input/your-dataset-name'
    BATCH_SIZE = 4
    LR = 0.0001
    EPOCHS = 10
    SEQ_LEN = 90 # [수정] 시퀀스 길이를 조절 가능한 파라미터로 설정
    BACKBONE_NAME = 'resnet50' # [수정] resnet18, resnet50 등 선택 가능
    HIDDEN_DIM = 256
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPGSequenceDataset(Dataset):
    def __init__(self, root_dir, num_frames=Config.SEQ_LEN, transform=None):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.transform = transform
        self.samples = []
        
        for label in ['Real', 'Fake']:
            label_path = os.path.join(root_dir, label)
            if not os.path.exists(label_path): continue
            
            for video_name in os.listdir(label_path):
                img_folder = os.path.join(label_path, video_name, 'CbCr_x15')
                if os.path.isdir(img_folder):
                    images = os.listdir(img_folder)
                    if len(images) >= self.num_frames:
                        self.samples.append((img_folder, 0 if label == 'Real' else 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder_path, label = self.samples[idx]
        images = sorted(os.listdir(folder_path))[:self.num_frames]
        
        frames = []
        for img_name in images:
            img = Image.open(os.path.join(folder_path, img_name)).convert('RGB')
            if self.transform:
                img = self.transform(img)
            frames.append(img)
        
        return torch.stack(frames), torch.tensor(label, dtype=torch.float32)

class PPGClassifier(nn.Module):
    def __init__(self, backbone_name=Config.BACKBONE_NAME, hidden_dim=Config.HIDDEN_DIM):
        super(PPGClassifier, self).__init__()
        
        # [신규] 선택한 백본에 따라 동적으로 특징 추출기 생성
        if backbone_name == 'resnet18':
            backbone = models.resnet18(pretrained=True)
            self.feature_dim = 512
        elif backbone_name == 'resnet50':
            backbone = models.resnet50(pretrained=True)
            self.feature_dim = 2048 # [주의] ResNet50 이상의 모델은 출력 차원이 2048임
        else:
            raise ValueError("Unsupported backbone")

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        
        # [수정] feature_dim을 백본에 맞춰 유연하게 설정
        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_dim=hidden_dim, num_layers=1, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        
        x = x.view(batch_size * seq_len, c, h, w)
        with torch.no_grad():
            x = self.feature_extractor(x)
        
        x = x.view(batch_size, seq_len, -1)
        
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        return self.fc(last_time_step).squeeze()

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    train_losses = []
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch+1} 완료! 평균 Loss: {epoch_loss:.4f}")
        
    return train_losses

if __name__ == "__main__":
    # [신규] 과적합 방지를 위한 데이터 증강(Augmentation) 추가
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5), # 좌우 반전 추가
        transforms.RandomRotation(10),           # 미세한 회전 추가
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = PPGSequenceDataset(Config.ROOT_DIR, transform=train_transform)
    
    if len(dataset) == 0:
        print("Error: No valid data found.")
    else:
        train_loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

        # [수정] Config에 정의된 백본으로 모델 초기화
        model = PPGClassifier(backbone_name=Config.BACKBONE_NAME).to(Config.DEVICE)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=Config.LR)

        print(f"학습 시작! Backbone: {Config.BACKBONE_NAME}, SeqLen: {Config.SEQ_LEN}")
        losses = train_model(model, train_loader, criterion, optimizer, Config.DEVICE, Config.EPOCHS)

        plt.plot(losses)
        plt.title(f"Training Loss ({Config.BACKBONE_NAME})")
        plt.show()

        torch.save(model.state_dict(), f"ppg_deepfake_{Config.BACKBONE_NAME}_v3.pth")
        print(f"모델 저장 완료: ppg_deepfake_{Config.BACKBONE_NAME}_v3.pth")
