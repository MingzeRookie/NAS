# CUDA_VISIBLE_DEVICES=0 python train.py --config exp/encoder_com/steatosis/max/uni/config.yaml &
# CUDA_VISIBLE_DEVICES=0 python train.py --config exp/encoder_com/inflam/max/uni/config.yaml &
# CUDA_VISIBLE_DEVICES=0 python train.py --config exp/encoder_com/ball/max/uni/config.yaml &
# CUDA_VISIBLE_DEVICES=1 python train.py --config exp/encoder_com/steatosis/max/conch/config.yaml &
# CUDA_VISIBLE_DEVICES=1 python train.py --config exp/encoder_com/inflam/max/conch/config.yaml &
# CUDA_VISIBLE_DEVICES=1 python train.py --config exp/encoder_com/ball/max/conch/config.yaml 
# --- exp:giga ---
# CUDA_VISIBLE_DEVICES=1 python train.py --config exp/encoder_com/steatosis/abmil/giga/config.yaml &
# CUDA_VISIBLE_DEVICES=1 python train.py --config exp/encoder_com/inflam/abmil/giga/config.yaml &
# CUDA_VISIBLE_DEVICES=1 python train.py --config exp/encoder_com/ball/abmil/giga/config.yaml &
# CUDA_VISIBLE_DEVICES=0 python train.py --config exp/encoder_com/steatosis/mean/giga/config.yaml &
# CUDA_VISIBLE_DEVICES=0 python train.py --config exp/encoder_com/inflam/mean/giga/config.yaml &
# CUDA_VISIBLE_DEVICES=0 python train.py --config exp/encoder_com/ball/mean/giga/config.yaml &
# CUDA_VISIBLE_DEVICES=0 python train.py --config exp/encoder_com/steatosis/max/giga/config.yaml &
# CUDA_VISIBLE_DEVICES=0 python train.py --config exp/encoder_com/inflam/max/giga/config.yaml &
# CUDA_VISIBLE_DEVICES=0 python train.py --config exp/encoder_com/ball/max/giga/config.yaml 

# --- exp:mamba ---
# CUDA_VISIBLE_DEVICES=0 python train.py --config exp/encoder_com/steatosis/mamba/uni/config.yaml 
# CUDA_VISIBLE_DEVICES=1 python train.py --config exp/encoder_com/steatosis/mamba/conch/config.yaml & 
# CUDA_VISIBLE_DEVICES=0 python train.py --config exp/encoder_com/steatosis/mamba/giga/config.yaml &
# CUDA_VISIBLE_DEVICES=1 python train.py --config exp/encoder_com/inflam/mamba/uni/config.yaml &
# CUDA_VISIBLE_DEVICES=0 python train.py --config exp/encoder_com/inflam/mamba/conch/config.yaml &
# CUDA_VISIBLE_DEVICES=0 python train.py --config exp/encoder_com/inflam/mamba/giga/config.yaml &
# CUDA_VISIBLE_DEVICES=1 python train.py --config exp/encoder_com/ball/mamba/uni/config.yaml &
# CUDA_VISIBLE_DEVICES=1 python train.py --config exp/encoder_com/ball/mamba/conch/config.yaml &
# CUDA_VISIBLE_DEVICES=1 python train.py --config exp/encoder_com/ball/mamba/giga/config.yaml 
# CUDA_VISIBLE_DEVICES=1 python train_graph.py --config exp/encoder_com/steatosis/patchgcn/uni/config.yaml 
# CUDA_VISIBLE_DEVICES=1 python train_graph.py --config exp/encoder_com/steatosis/patchgcn/giga/config.yaml 


# wikg
CUDA_VISIBLE_DEVICES=1 python train.py --config exp/encoder_com/steatosis/mean/nase/config.yaml 
# CUDA_VISIBLE_DEVICES=0 python train.py --config exp/encoder_com/steatosis/wikg/conch/config.yaml 
# CUDA_VISIBLE_DEVICES=0 python train.py --config exp/encoder_com/steatosis/wikg/uni/config.yaml &
# CUDA_VISIBLE_DEVICES=1 python train.py --config exp/encoder_com/ball/wikg/uni/config.yaml &
# CUDA_VISIBLE_DEVICES=0 python train.py --config exp/encoder_com/ball/wikg/conch/config.yaml &
# CUDA_VISIBLE_DEVICES=0 python train.py --config exp/encoder_com/inflam/max/uni/config.yaml 

# CUDA_VISIBLE_DEVICES=1 python train.py --config exp/encoder_com/nas/mean/uni/config.yaml &
# CUDA_VISIBLE_DEVICES=1 python train.py --config exp/encoder_com/nas/mean/giga/config.yaml &
# CUDA_VISIBLE_DEVICES=1 python train.py --config exp/encoder_com/nas/mean/conch/config.yaml &
# CUDA_VISIBLE_DEVICES=1 python train.py --config exp/encoder_com/nas/max/uni/config.yaml &
# CUDA_VISIBLE_DEVICES=1 python train.py --config exp/encoder_com/nas/max/giga/config.yaml &
# CUDA_VISIBLE_DEVICES=1 python train.py --config exp/encoder_com/nas/max/conch/config.yaml &
# CUDA_VISIBLE_DEVICES=2 python train.py --config exp/encoder_com/nas/abmil/uni/config.yaml &
# CUDA_VISIBLE_DEVICES=2 python train.py --config exp/encoder_com/nas/abmil/giga/config.yaml &
# CUDA_VISIBLE_DEVICES=2 python train.py --config exp/encoder_com/nas/abmil/conch/config.yaml 
# CUDA_VISIBLE_DEVICES=0 python train.py --config exp/encoder_com/nas/wikg/uni/config.yaml &
# CUDA_VISIBLE_DEVICES=1 python train.py --config exp/encoder_com/nas/wikg/giga/config.yaml &
# CUDA_VISIBLE_DEVICES=2 python train.py --config exp/encoder_com/nas/wikg/conch/config.yaml 
wait
# CUDA_VISIBLE_DEVICES=2 python train_nasc.py --config exp/mil/nasc/steatosis/diff/config.yaml 
# CUDA_VISIBLE_DEVICES=0 python train_nasc.py --config exp/mil/nasc/steatosis/diff_soft/config.yaml 
# CUDA_VISIBLE_DEVICES=0 python train_nasc.py --config exp/mil/nasc/steatosis/diff_soft_0.2/config.yaml &
# CUDA_VISIBLE_DEVICES=1 python train_nasc.py --config exp/mil/nasc/steatosis/diff_soft_0.3/config.yaml &
# CUDA_VISIBLE_DEVICES=2 python train_nasc.py --config exp/mil/nasc/steatosis/diff_soft_0.4/config.yaml &
# CUDA_VISIBLE_DEVICES=3 python train_nasc.py --config exp/mil/nasc/steatosis/diff_soft_1.25/config.yaml &
# CUDA_VISIBLE_DEVICES=4 python train_nasc.py --config exp/mil/nasc/steatosis/diff_soft_1.5/config.yaml 
# wait