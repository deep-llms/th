#1
#check-train-configs

# Check training args saved in checkpoints
echo "=== BASELINE config ==="
python3 -c "
import json
with open('/opt/dlami/nvme/smoke_test_outputs/baseline/checkpoint-6500/trainer_state.json') as f:
    s = json.load(f)
print('total steps:', s.get('global_step'))
print('epoch:', s.get('epoch'))
"
cat /opt/dlami/nvme/smoke_test_outputs/baseline/training_args.bin 2>/dev/null || echo "no training_args.bin"
python3 -c "
import torch
args = torch.load('/opt/dlami/nvme/smoke_test_outputs/baseline/training_args.bin', weights_only=False)
for k in ['learning_rate','per_device_train_batch_size','gradient_accumulation_steps','num_train_epochs','lr_scheduler_type','warmup_steps','bf16','seed']:
    print(f'{k}: {getattr(args, k, \"N/A\")}')
" 2>/dev/null || echo "could not load training_args.bin"

echo ""
echo "=== S3 a015 config ==="
python3 -c "
import torch
args = torch.load('/opt/dlami/nvme/smoke_test_outputs/S3_a015/training_args.bin', weights_only=False)
for k in ['learning_rate','per_device_train_batch_size','gradient_accumulation_steps','num_train_epochs','lr_scheduler_type','warmup_steps','bf16','seed']:
    print(f'{k}: {getattr(args, k, \"N/A\")}')
" 2>/dev/null || echo "could not load training_args.bin"

echo ""
echo "=== S3 a02 config ==="
python3 -c "
import torch
args = torch.load('/opt/dlami/nvme/smoke_test_outputs/S3_a02/training_args.bin', weights_only=False)
for k in ['learning_rate','per_device_train_batch_size','gradient_accumulation_steps','num_train_epochs','lr_scheduler_type','warmup_steps','bf16','seed']:
    print(f'{k}: {getattr(args, k, \"N/A\")}')
" 2>/dev/null || echo "could not load training_args.bin"

# Check embhub config
echo ""
echo "=== S3 a015 embhub config ==="
cat /opt/dlami/nvme/smoke_test_outputs/S3_a015/checkpoint-6500/embhub_config.json 2>/dev/null || echo "no embhub_config.json"

echo ""
echo "=== S3 a02 embhub config ==="
cat /opt/dlami/nvme/smoke_test_outputs/S3_a02/checkpoint-6500/embhub_config.json 2>/dev/null || echo "no embhub_config.json"

# Check if there's a run command / script logged
echo ""
echo "=== Check for launch commands ==="
ls /opt/dlami/nvme/smoke_test_outputs/baseline/*.log 2>/dev/null | head -3
ls /opt/dlami/nvme/smoke_test_outputs/S3_a015/*.log 2>/dev/null | head -3
ls /opt/dlami/nvme/smoke_test_outputs/S3_a02/*.log 2>/dev/null | head -3

echo ""
echo "=== Baseline: check if --no_embhub flag was used ==="
head -5 /opt/dlami/nvme/smoke_test_outputs/baseline/checkpoint-6500/config.json 2>/dev/null

echo ""
echo "=== S3 a015: alpha value in embhub weights ==="
python3 -c "
import torch, os
p = '/opt/dlami/nvme/smoke_test_outputs/S3_a015/checkpoint-6500/embhub.pt'
if os.path.exists(p):
    w = torch.load(p, map_location='cpu', weights_only=True)
    print('keys:', list(w.keys()))
    if 'log_logit_scale' in w:
        import math
        print('logit_scale:', math.exp(w['log_logit_scale'].item()))
else:
    print('no embhub.pt')
"

echo ""
echo "=== train_config.json (saved by our code) ==="
for ARM in baseline S3_a015 S3_a02; do
  echo "--- $ARM ---"
  python3 -c "
import json
with open('/opt/dlami/nvme/smoke_test_outputs/${ARM}/checkpoint-6500/train_config.json') as f:
    d = json.load(f)
print('no_embhub:', d['embhub']['no_embhub'])
print('alpha:', d['embhub']['alpha'])
print('scale_lr_mult:', d['embhub'].get('scale_lr_mult', 'N/A'))
print('scale_no_wd:', d['embhub'].get('scale_no_wd', 'N/A'))
print('scale_init:', d['embhub'].get('scale_init', 'N/A'))
print('lr:', d['training']['learning_rate'])
print('batch:', d['training']['per_device_train_batch_size'])
print('grad_accum:', d['training']['gradient_accumulation_steps'])
print('seed:', d['training']['seed'])
" 2>/dev/null || echo "  no train_config.json"
done

echo ""
echo "=== S3 a02: alpha value in embhub weights ==="
python3 -c "
import torch, os
p = '/opt/dlami/nvme/smoke_test_outputs/S3_a02/checkpoint-6500/embhub.pt'
if os.path.exists(p):
    w = torch.load(p, map_location='cpu', weights_only=True)
    print('keys:', list(w.keys()))
    if 'log_logit_scale' in w:
        import math
        print('logit_scale:', math.exp(w['log_logit_scale'].item()))
else:
    print('no embhub.pt')
"
