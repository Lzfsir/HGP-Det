from repvgg import repvgg_model_convert, create_RepVGG_A0
import torch

train_model = create_RepVGG_A0(deploy=False)
train_model.load_state_dict(torch.load('RepVGG-A0-train.pth'))          # or train from scratch
# do whatever you want with train_model
deploy_model = repvgg_model_convert(train_model, save_path='RepVGG-A0-deploy.pth')
print(deploy_model)
# do whatever you want with deploy_model
