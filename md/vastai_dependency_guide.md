# ComfyUIクラウド環境における依存関係見失いエラーの全容と解決策

このドキュメントは、Vast.aiやRunPodなどのクラウド環境でSeedVR2（及びその他の重度な外部依存を持つカスタムノード）を動作させる際に発生する `ModuleNotFoundError` エラーについて、その本質的な原因から具体的なコードレベルの修正内容までを完全解説したものです。

---

## 1. エラーの内容と発生のメカニズム

**【発生したエラー】**

```python
ModuleNotFoundError: No module named 'diffusers'
ModuleNotFoundError: No module named 'rotary_embedding_torch'
```

Vast.aiのターミナル上で `pip install diffusers` などを実行して**「Successfully installed」**と表示されているにもかかわらず、ComfyUIでSeedVR2がロードされる際に上記のエラーが発生して強制終了します。

### 本質的な原因（2つの要因の衝突）

このエラーは、以下の「ComfyUIの思想」と「クラウド環境の仕様」という2つの要因が最悪の形で噛み合った結果発生します。

#### 要因A：ComfyUIの「反・巨大ライブラリ」思想

画像の生成AI界隈において、`diffusers` や `transformers`、`accelerate` などのライブラリは**あって当然の超・標準ライブラリ**です（Automatic1111等もこれらを利用しています）。
しかし、ComfyUIは「不要な抽象化や巨大なライブラリに頼らず、PyTorchのみで軽量・最速な計算グラフを回す」という思想で作られています。そのため、**あえて `diffusers` 等を本体の必須環境から徹底的に排除しています。**
これにより、クラウド側が用意した「クリーンなComfyUI環境」には、一般には標準とされるライブラリ群がインストールされていません。

#### 要因B：クラウド環境での「見えない複数Python環境」

Vast.aiによるComfyUIテンプレートでは、通常以下のように環境が分断されています。

1. **ターミナル（ユーザー側）**: `/usr/bin/python` 等のシステムPython
2. **ComfyUI実行プロセス**: `/workspace/ComfyUI/venv/bin/python` のような隠されたPython仮想環境 (venv)

ユーザーがターミナルを開いて叩く `pip install` は常に「システム側」へ向けて実行されます。しかし、実際にノードをロードして動かしているのは「ComfyUI実行プロセス」側のPythonです。ComfyUIの仮想環境には手が出せず、**UI上からはインストール先を指定する術が存在しません。**

---

## 2. 修正したファイルとコードの詳細

この設計上の罠を力技で突破するため、SeedVR2の起動エントリポイントである `__init__.py` を大幅に書き換えました。

**修正対象ファイル:** `/ComfyUI/custom_nodes/ComfyUI-SeedVR2_VideoUpscaler/__init__.py`

### 実装したコード（抜粋）

```python
import sys
import subprocess

def ensure_package(package_name, import_name=None):
    if import_name is None:
        import_name = package_name.split("\>")[0].split("=")[0].split("<")[0]
    
    try:
        # まず実際にimportを試行し、機能するか確認する
        __import__(import_name)
        return  # 既にインストールされていればスキップ
    except (ImportError, ModuleNotFoundError):
        pass
    
    # パッケージが見つからなかった場合の強行インストール処理
    print("\n" + "="*80)
    print(f"SeedVR2: '{import_name}' module not found.")
    print(f"SeedVR2: Current Python executable: {sys.executable}")
    print(f"SeedVR2: Attempting automatic installation of {package_name}...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
        print(f"SeedVR2: Successfully installed {package_name}")
    except Exception as e:
        print(f"SeedVR2: Auto-installation failed: {e}")
    print("="*80 + "\n")

# requirements.txtのすべての依存関係を自動インストールの対象に登録
_REQUIRED_PACKAGES = [
    ("safetensors", None),
    ("tqdm", None),
    ("omegaconf>=2.3.0", "omegaconf"),
    ("diffusers>=0.33.1", "diffusers"),
    ("transformers", None),
    ("accelerate", None),
    ("peft>=0.17.0", "peft"),
    ("rotary_embedding_torch>=0.5.3", "rotary_embedding_torch"),
    ("opencv-python", "cv2"), # pip名とimport名が異なるもののマッピング
    ("gguf", None),
]

for pkg, imp in _REQUIRED_PACKAGES:
    ensure_package(pkg, imp)
```

---

## 3. コードの意味とその効果（なぜこれで解決するのか）

追加されたコードがどのように「環境のすれ違い」を解決しているのかを解説します。

### ① `sys.executable` による「現在位置」の特定と実行

このコードの最も重要なコアは `sys.executable` にあります。
`sys.executable` とは、**「現在このPythonスクリプトを実行しているPython自体のパス（＝裏で動いているComfyUI用のvenvのパス）」**を指します。

`subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])`
を実行することで、外側のターミナル環境を完全に無視し、**「ComfyUIを動かしているまさにその仮想環境に対して、内側から直接 pip install コマンドを発行」**します。これにより、インストール先の指定不可能問題を突破できます。

### ② `try / __import__()` による確実な欠損チェック

Python標準の `importlib.find_spec()` は、パッケージが破損していたり、不完全な状態でも「存在する」と誤判定を返すことがあります。
そのため、実際に `__import__()` を試みて、それが `ModuleNotFoundError` で落ちた場合にのみインストールを走らせることで、堅牢性を担保しています。

### ③ インポート名とpip名（パッケージ名）の不一致対応

Python系のパッケージには、`pip install xxx` と打つ名前と、コード内で `import yyy` と記述する名前が違うものが多々あります（例: `pip install opencv-python` だがインポート時は `cv2`）。
変数 `_REQUIRED_PACKAGES` にて、「pipインストール名」と「実際にインポート可能かを探る名前」を明示的にマッピング・分離することで、無限に再インストールが繰り返されるバグを防止しています。

### まとめ

この修正により、Vast.aiやRunPodにノードを `git clone` で直接手動導入した場合でも、ComfyUI起動時に**「自身の動いている環境を自らスキャンし、足りないものは自分自身に直接手当て（インストール）する」自己修復機能**が働くようになりました。ユーザーがターミナルでパスに思い悩む必要はなくなりました。
