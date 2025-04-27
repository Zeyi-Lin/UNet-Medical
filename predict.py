import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from net import UNet
import numpy as np
import os

def load_model(model_path='best_model.pth', device='cuda'):
    """加载训练好的模型"""
    try:
        # 检查文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        model = UNet(n_filters=32).to(device)
        # 添加weights_only=True来避免警告
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def preprocess_image(image_path):
    """预处理输入图像"""
    # 读取原始图像
    image = Image.open(image_path).convert('RGB')
    
    # 保存调整大小后的原始图像用于显示
    display_image = image.resize((256, 256), Image.Resampling.BILINEAR)
    
    # 模型输入的预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0), display_image

def predict_mask(model, image_tensor, device='cuda', threshold=0.5):
    """预测分割掩码"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        prediction = model(image_tensor)
        prediction = (prediction > threshold).float()
    return prediction

def visualize_result(original_image, predicted_mask):
    """可视化预测结果"""
    plt.figure(figsize=(12, 6))
    plt.suptitle('Predictions')
    
    # 显示原始图像
    plt.subplot(131)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')
    
    # 显示预测掩码
    plt.subplot(132)
    plt.imshow(predicted_mask.squeeze(), cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')
    
    # 显示叠加结果
    plt.subplot(133)
    plt.imshow(np.array(original_image))  # 转换为numpy数组
    plt.imshow(predicted_mask.squeeze(), cmap='Reds', alpha=0.3)
    plt.title('Overlay')
    plt.axis('off')
        
    plt.tight_layout()
    plt.savefig('./predictions.png')
    print("Visualization saved as predictions.png")

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # 加载模型
        model_path = "./best_model.pth"  # 确保这个路径是正确的
        print(f"Attempting to load model from: {model_path}")
        model = load_model(model_path, device)
        
        # 处理单张图像
        image_path = "dataset/test/27_jpg.rf.b2a2b9811786cc32a23c46c560f04d07.jpg"
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found at {image_path}")
            
        print(f"Processing image: {image_path}")
        image_tensor, original_image = preprocess_image(image_path)
        
        # 预测
        predicted_mask = predict_mask(model, image_tensor, device)
        
        # 将预测结果转回CPU并转换为numpy数组
        predicted_mask = predicted_mask.cpu().numpy()
        
        # 可视化结果
        print("Generating visualization...")
        visualize_result(original_image, predicted_mask)
        print("Results saved to predictions.png")
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise

if __name__ == '__main__':
    main()
