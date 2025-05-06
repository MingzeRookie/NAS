from dinov2.data.datasets import NASH


for split in NASH.Split:
    # print(split)
    dataset = NASH(split=split,root='/remote-home/share/lisj/Workspace/SOTA_NAS/datasets/core/patches',extra='/remote-home/share/lisj/Workspace/SOTA_NAS/datasets/core/extra')
    break

dataset.dump_extra()