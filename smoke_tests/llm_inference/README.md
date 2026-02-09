推論モジュールの動作チェック（Docker コンテナ内で実行）

## 事前準備（必要な場合のみ）
### モデルの取得例
```sh
hf download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B
hf download openai/gpt-oss-20b --local-dir models/gpt-oss-20b
```

### 環境変数
- OpenAI API を使う場合
  - `OPENAI_API_KEY`
- OpenAI API 互換サーバを使う場合
  - `BASE_URL`（例: `http://localhost:8000/v1`）
  - `API_KEY`（認証が不要ならダミー値でも可）

## 実行方法
以下は `smoke_tests/llm_inference` ディレクトリで実行する想定です。

```sh
# OpenAI API の動作チェック
python3 test_openai_api.py

# vLLM の Offline Inference の動作チェック
python3 test_vllm_offline.py
python3 test_vllm_offline_gptoss.py

# vLLM の OpenAI API 互換サーバの動作チェック
chmod +x test_vllm_serve.sh
./test_vllm_serve.sh

# SGLang の Offline Inference の動作チェック
python3 test_sglang_offline.py

# SGLang の OpenAI API 互換サーバの動作チェック
chmod +x test_sglang_serve.sh
./test_sglang_serve.sh

# OpenAI API 互換サーバへの直接疎通チェック（vLLM/SGLang 共通）
python3 test_vllm_openai_api_compatible.py
python3 test_sglang_openai_api_compatible.py
```

## 各テストの概要
- `test_openai_api.py`
  - OpenAI API に対して短文/会話/バッチリクエストを実行します。
- `test_vllm_offline.py` / `test_vllm_offline_gptoss.py`
  - vLLM のオフライン推論でモデルパスを指定して推論します。
  - それぞれ `Qwen3-0.6B` と `gpt-oss-20b` を想定。
- `test_vllm_serve.sh`
  - vLLM の OpenAI API 互換サーバを起動し、疎通確認します。
  - `BASE_URL` と `API_KEY` を指定すると外部のサーバにも接続できます。
- `test_sglang_offline.py`
  - SGLang のオフライン推論でモデルパスを指定して推論します。
- `test_sglang_serve.sh`
  - SGLang の OpenAI API 互換サーバを起動し、疎通確認します。
- `test_vllm_openai_api_compatible.py` / `test_sglang_openai_api_compatible.py`
  - 起動済みの OpenAI API 互換サーバに対して、`BASE_URL` と `API_KEY` を使って疎通確認します。

## トラブルシューティング
- `RuntimeError: OPENAI_API_KEY is not set`
  - `OPENAI_API_KEY` を設定してください。
- `RuntimeError: BASE_URL is not set`
  - `BASE_URL` を設定してから実行してください。
- モデルが見つからない場合
  - `models/` の配置場所を確認し、コンテナ内で `/workspace/models/...` が参照できるようにマウントしてください。
