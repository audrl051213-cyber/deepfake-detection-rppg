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

# 1. [Dataset] 데이터를 어떻게 읽어올지 정의하는 클래스
class PPGSequenceDataset(Dataset):
    def __init__(self, root_dir, num_frames=90, transform=None):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.transform = transform
        self.samples = []
        
        # 폴더 구조: root_dir / [Real or Fake] / [영상이름] / CbCr_x15 / [이미지들]
        for label in ['Real', 'Fake']:
            label_path = os.path.join(root_dir, label)
            if not os.path.exists(label_path): continue
            
            for video_name in os.listdir(label_path):
                img_folder = os.path.join(label_path, video_name, 'CbCr_x15')
                if os.path.isdir(img_folder):
                    # [수정 제안 3 반영] 이미지 개수가 충분한지 미리 체크
                    images = os.listdir(img_folder)
                    if len(images) >= self.num_frames:
                        self.samples.append((img_folder, 0 if label == 'Real' else 1))
                    else:
                        print(f"Warning: {img_folder} has only {len(images)} frames. Skipping...")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder_path, label = self.samples[idx]
        # 시간 순서 보장을 위한 정렬
        images = sorted(os.listdir(folder_path))[:self.num_frames]
        
        frames = []
        for img_name in images:
            img = Image.open(os.path.join(folder_path, img_name)).convert('RGB')
            if self.transform:
                img = self.transform(img)
            frames.append(img)
        
        return torch.stack(frames), torch.tensor(label, dtype=torch.float32)

# 2. [Model] 신경망 구조 정의 (CNN + LSTM)
class PPGClassifier(nn.Module):
    def __init__(self, hidden_dim=256):
        super(PPGClassifier, self).__init__()
        # Backbone: ResNet18
        backbone = models.resnet18(pretrained=True)
        # 마지막 FC 레이어 제외한 특징 추출기
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        
        # [수정 제안 1 관련] 백본 가중치 고정 (필요시 활성화)
        # for param in self.feature_extractor.parameters():
        #     param.requires_grad = False
            
        self.lstm = nn.LSTM(input_size=512, hidden_dim=hidden_dim, num_layers=1, batch_first=True)
        
        # [수정 제안 2 반영] BCEWithLogitsLoss 사용을 위해 Sigmoid 제거
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
            # Sigmoid 제거됨
        )

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        
        # 특징 추출 단계
        # (Batch * Seq, C, H, W) 형태로 펼쳐서 CNN 통과
        x = x.view(batch_size * seq_len, c, h, w)
        with torch.no_grad(): # [수정 제안 1 반영] 특징 추출 시 메모리 절약
            x = self.feature_extractor(x)
        
        x = x.view(batch_size, seq_len, -1)
        
        # LSTM 및 분류기 통과
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        return self.fc(last_time_step).squeeze()

# 3. [Training] 학습 함수 정의
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
            # [수정 제안 2 관련] 내부에서 Sigmoid 연산이 포함된 Loss 사용
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
    # 설정값
    ROOT_DIR = '/kaggle/input/your-dataset-name'
    BATCH_SIZE = 4
    LR = 0.0001
    EPOCHS = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = PPGSequenceDataset(ROOT_DIR, transform=transform)
    if len(dataset) == 0:
        print("Error: No valid data found. Check your ROOT_DIR and folder structure.")
    else:
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        model = PPGClassifier().to(DEVICE)
        
        # [수정 제안 2 반영] BCEWithLogitsLoss 사용
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)

        print(f"학습을 시작합니다! 장치: {DEVICE}")
        losses = train_model(model, train_loader, criterion, optimizer, DEVICE, EPOCHS)

        plt.plot(losses)
        plt.title("Training Loss")
        plt.show()

        torch.save(model.state_dict(), "ppg_deepfake_model_v2.pth")
        print("모델 저장 완료! (Version 2)")
