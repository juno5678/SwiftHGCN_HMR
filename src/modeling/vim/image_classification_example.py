import torch
import sys
from models_mamba import VisionMamba
import time

MODEL_FILE = "./checkpoint/vim_s_midclstok_ft_81p6acc.pth"
print(MODEL_FILE)

model = VisionMamba(
    patch_size=16,
    stride=8,
    embed_dim=384,
    depth=24,
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    final_pool_type='mean',
    if_abs_pos_embed=True,
    if_rope=False,
    if_rope_residual=False,
    bimamba_type="v2",
    if_cls_token=True,
    if_devide_out=True,
    use_middle_cls_token=True,
    num_classes=1000,
    drop_rate=0.0,
    drop_path_rate=0.1,
    drop_block_rate=None,
    img_size=224,
)

checkpoint = torch.load(str(MODEL_FILE), map_location='cpu')
# Important: make sure the values of this match what's used to instantiate the VisionMamba class.
# If not, loading the checkpoint will fail.
# print(checkpoint["args"])

model.load_state_dict(checkpoint["model"])


model.eval()
model.to("cuda")

from PIL import Image
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

test_image = Image.open("./samples/test.png")
test_image = test_image.resize((224, 224))
image_as_tensor = transforms.ToTensor()(test_image)
normalized_tensor = transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(image_as_tensor)

name = 'vim'

t_cnt = 0.0
with torch.no_grad():
    x = normalized_tensor.unsqueeze(0).cuda()
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    pred = model(x)
    pred = model(x)
    torch.cuda.synchronize()
    start_ts = time.time()

    for i in range(100):
        pred = model(x)
    torch.cuda.synchronize()
    end_ts = time.time()

t_cnt = end_ts - start_ts  # t_cnt + (end_ts-start_ts)
print("=======================================")
print("Model Name: " + name)
print("FPS: %f" % (100 / t_cnt))
print("=======================================")

# Note: the returned label can be verified with https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/
print(pred.argmax())

# 获取模型参数
params = model.parameters()
# 计算参数量
num_params = 0
for param in params:
    num_params += torch.prod(torch.tensor(param.size()))
print('模型参数量：', num_params)



from torchsummary import summary
# 创建模型对象
# 计算模型计算量
# summary(model, input_size=(3, 224, 224))
