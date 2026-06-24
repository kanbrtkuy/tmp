# RunPod Setup

## Fresh Pod Checklist

1. Clone this repo.
2. Create Python environments or install editable package.
3. Inject Hugging Face token through SSH stdin.
4. Download/cache models.
5. Validate configs and run smoke tests.

## Token Injection

Do not write the real token into GitHub, docs, shell history, or logs.

Local private token file:

```bash
mkdir -p ~/.safechain_secrets
chmod 700 ~/.safechain_secrets
printf '%s\n' '<PASTE_HF_TOKEN_HERE>' > ~/.safechain_secrets/hf_token
chmod 600 ~/.safechain_secrets/hf_token
```

Install on a pod:

```bash
POD_ALIAS=<new-runpod-ssh-alias>
ssh "${POD_ALIAS}" 'mkdir -p /workspace/secrets /workspace/hf_cache /root/.cache/huggingface && chmod 700 /workspace/secrets /root/.cache/huggingface && umask 077 && TOKEN=$(cat) && printf "export HF_TOKEN=%s\nexport HUGGING_FACE_HUB_TOKEN=%s\nexport HF_HOME=/workspace/hf_cache\n" "$TOKEN" "$TOKEN" > /workspace/secrets/hf.env && printf "%s" "$TOKEN" > /root/.cache/huggingface/token' < ~/.safechain_secrets/hf_token
```

Load for remote commands:

```bash
source /workspace/secrets/hf.env
```

## 4×A100 80GB Runtime

Use:

```yaml
defaults:
  - configs/runtime/a100_4x.yaml
```

This is the preferred setup for DeepSeek 8B full SFT and larger hidden-state /
judge workloads.
