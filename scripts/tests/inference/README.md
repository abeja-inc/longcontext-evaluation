推論モジュールの動作チェック（コンテナ内で実行）

```sh
# OpenAI API の動作チェック
python3 test_openai_api.py

# vLLM の Offline Inference の動作チェック
python3 test_vllm_offline_inference.py
python3 test_vllm_offline_inference_gptoss.py

# vLLM の OpenAI API 互換サーバの動作チェック
chmod +x test_vllm_serve.py
./test_vllm_serve.sh

# SGLang の Offline Inference の動作チェック
python3 test_sglang_offline.py
```
