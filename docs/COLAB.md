# Colabでの実行

## 共通の準備

1. この公開リポジトリの `notebooks/StreamCut.ipynb` をColabで開く。
2. 使用するGoogleアカウントをColabとDriveで選ぶ。ソースコードにはアカウント・OAuth情報を保存しない。
3. 初回だけDriveをマウントし、設定セルの `project_name` を新しい名前にする。すべてのワーカーで同じ名前を使う。
4. Colab Secretsに `STREAMCUT_API_KEY` を保存し、coordinator/singleのノートブックにアクセスを許可する。GPU/renderワーカーへAPIキーは渡さない。
5. `llm_model` に自分が利用するモデルIDを明示する。既定モデルや有料fallbackはない。

ノートブックはタグ `v0.1.1` を使う。全台で同じ版を指定する。uvでPython 3.12を準備し、依存lockで独立した環境を作る。初回セットアップはPython・パッケージ・モデルの取得を含み、数分以上かかる場合がある。ランタイムごとの環境構築時間も `Colab setup seconds` として表示する。

## 最初はGPU1台

GPUランタイムを選び、`role=single`、`gpu_workers=1`、`render_workers=1` を指定する。入力にDrive上の動画パスかアーカイブURLを入れ、実行する。モデル・Drive・FFmpegの実動作を確認するには、まず短い実素材を使う。

## GPU4台＋CPU4台

| ランタイム | role | rank | 台数 |
|---|---|---|---|
| CPU coordinator | coordinator | 0 | 1 |
| GPU worker | gpu | 0,1,2,3 | 4 |
| CPU render worker | render | 1,2,3 | 3 |

coordinatorで `gpu_workers=4`、`render_workers=4` を指定する。追加CPU workerのrank 0は起動しない。coordinator自身がCPU描画rank 0を担当する。

各ランタイムの実行ボタンを押すと、ワーカーは同じ共有フォルダで仕事を待つ。coordinatorが取得・解析タスク作成・選定・仕上げタスク作成を進める。GPUはASRの後、選定が終わるまで待ち、採用区間だけalignmentして終了する。CPUは描画タスクが来るまで待つ。

**Colabが8つのランタイムを同時に割り当てられる保証はない。** 使える台数を起動前に確認し、1〜4で設定する。アカウントやアクセラレーターを自動変更して利用制限を回避する処理はない。

共有Driveフォルダは全台で同じ絶対パスにマウントする。音声パートは推論前に各GPUのローカル一時領域へコピーする。動画は共有する元ファイルをCPUから読み、Clipごとに全動画を複製しない。Driveの読み込みが詰まる場合はCPU台数を減らす。

## コマンドでの対応

coordinator:

```bash
streamcut run /content/drive/MyDrive/Inputs/source.mp4 \
  --project /content/drive/MyDrive/StreamCut/job-001 \
  --model YOUR_MODEL_ID --distributed --gpu-workers 4 --render-workers 4
```

各GPU（rankだけ変える）:

```bash
streamcut worker /content/drive/MyDrive/StreamCut/job-001 --role gpu --rank 0
```

追加CPU（rankは1〜3）:

```bash
streamcut worker /content/drive/MyDrive/StreamCut/job-001 --role render --rank 1
```

## 終了・再開

ノートブックの `release_runtime` を有効にすると、処理セルの終了時に自分自身のランタイムだけを解放する。無効なら手動で「接続解除してランタイムを削除」を実行する。他のノートブックのランタイムは停止しない。セットアップセルで失敗した場合も、不要なランタイムは手動で停止する。

失敗時はcoordinatorと全ワーカーを停止し、原因を直して、同じ設定・同じrankで起動する。coordinatorには `resume=true` を指定する。成功したASRやalignmentはreceiptから再利用する。

同じフォルダへcoordinatorを二重起動しない。強制終了で `.coordinator.lock` が残った場合は、元のcoordinatorが停止していることを確認してから、そのファイルだけを削除する。Drive上の排他的ファイル作成は分散DBのロックではないため、起動数を利用者側でも管理する。

モデルや字幕設定を変更した場合は新しい作業フォルダを使う。作業フォルダには動画・本文・モデル応答・ログが入るため、公開共有・Gitコミットしない。
