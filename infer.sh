module load stack/2024-06 eth_proxy gcc/12.2.0 cuda/11.8.0
python infer.py big --resume pretrained/model_fp16_fixrot.safetensors --workspace workspace_test --test_path data_test 

