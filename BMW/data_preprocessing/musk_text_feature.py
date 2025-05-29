import torch
from timm.models import create_model
import sys, os
from transformers import XLMRobertaTokenizer
import torch.nn.functional as F

musk_lib_path = "/remote-home/share/lisj/Workspace/SOTA_NAS/encoder/musk/MUSK-main"
if musk_lib_path not in sys.path:
    sys.path.insert(0, musk_lib_path)

try:
    from musk import utils, modeling
except ImportError:
    sys.exit(1)

device = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cuda:0" if torch.cuda.is_available() else "cpu")

model_config = "musk_large_patch16_384"
local_ckpt = "/remote-home/share/lisj/Workspace/SOTA_NAS/encoder/musk/checkpoint/model.safetensors"
tokenizer_path = "/remote-home/share/lisj/Workspace/SOTA_NAS/encoder/musk/MUSK-main/musk/models/tokenizer.spm"
save_path = "/remote-home/share/lisj/Workspace/SOTA_NAS/datasets/core/MUSK-feature/MUSK-text-feature/tissue_text_feature.pt"

max_token_len = 100

prompts = [
        """A whole slide image of lobular inflammation in non-alcoholic steatohepatitis
    at high resolution typically displays small clusters of inflammatory cells, 
    primarily lymphocytes, scattered irregularly throughout the hepatic lobules. 
    These infiltrates are not confined to portal areas but instead intersperse between hepatocytes, 
    often surrounding ballooned or necrotic hepatocytes. The surrounding parenchyma may show mild disarray, 
    with occasional microvesicular fat droplets and subtle sinusoidal dilation, 
    while the background includes zones of hepatocellular degeneration and focal hepatocyte dropout.
""", 
]

model = create_model(model_config).eval()
try:
    utils.load_model_and_may_interpolate(local_ckpt, model, 'model|module', '')
    model.to(device, dtype=torch.float16)
    model.eval()
except FileNotFoundError:
    sys.exit(1)
except Exception as e:
    sys.exit(1)

try:
    tokenizer = XLMRobertaTokenizer(tokenizer_path)
    vocab_size = tokenizer.vocab_size
except Exception as e:
    sys.exit(1)

all_text_ids = []
all_paddings = []
try:
    for i, txt in enumerate(prompts):
        txt_ids, pad = utils.xlm_tokenizer(txt, tokenizer, max_len=max_token_len)

        max_id = max(txt_ids)
        min_id = min(txt_ids)
        if max_id >= vocab_size or min_id < 0:
            pass  # 原警告提示已移除

        all_text_ids.append(torch.tensor(txt_ids).unsqueeze(0))
        all_paddings.append(torch.tensor(pad).unsqueeze(0))

    if not all_text_ids:
        sys.exit(1)

    batch_text_ids = torch.cat(all_text_ids).to(device)
    batch_paddings = torch.cat(all_paddings).to(device)

except Exception as e:
    sys.exit(1)

try:
    with torch.inference_mode():
        text_embeddings_per_prompt = model(
            text_description=batch_text_ids,
            padding_mask=batch_paddings,
            with_head=True,
            out_norm=True
        )[1]
except Exception as e:
    if "CUDA error: device-side assert triggered" in str(e):
        pass  # 原CUDA错误提示已移除
    sys.exit(1)

averaged_embedding = text_embeddings_per_prompt.mean(dim=0, keepdim=True)
final_feature = F.normalize(averaged_embedding, dim=-1)

feature_to_save = final_feature.cpu().half()

output_dir = os.path.dirname(save_path)
os.makedirs(output_dir, exist_ok=True)

try:
    torch.save(feature_to_save, save_path)
except Exception as e:
    sys.exit(1)