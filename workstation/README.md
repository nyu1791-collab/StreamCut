# 新しいPCでの導入・移行

2026-10-31。Core i7 / メモリ16GB / 有線LANを想定する。GPU・Windowsの版・空きディスク・CapCutの版は未確認。Windowsネイティブの編集を基本にし、Dockerを使う場合だけDocker Desktop/WSL2を追加する。

## 最初に用意するもの

- Python 3.12、Git。uvを使うプロジェクトはuvも導入する。MoneyPrinterはuvが独立したPython 3.11環境を用意する。
- 動画処理にはFFmpeg/FFprobe（ASS字幕対応ビルド）、日本語フォント。CapCutはPCへ本人のProアカウントで導入する。Pro契約だけで外部投稿APIや非公式ドラフト形式の互換性が保証されるわけではない。
- ソースとデータを分ける。例: `C:\Automation\repos` と `C:\AutomationData`。各リポジトリはそれぞれの `.venv` を持ち、動画・DB・秘密をGitに入れない。Linuxの `.venv` をWindowsへコピーして使わない。

必要ならWindowsのwingetで `Python.Python.3.12`、`Git.Git`、`astral-sh.uv`、`Gyan.FFmpeg`、`OpenJS.NodeJS.LTS` を本人の端末へ導入する。Dockerは任意。今回ここからPCへインストールしたわけではない。

## このリポジトリだけ導入する

```powershell
py -3.12 workstation/manage.py doctor
# まず導入コマンドを見る。モデル/API/制作は起動しない。
py -3.12 workstation/manage.py setup
# 表示されたこのリポジトリの環境だけを実際に導入する
py -3.12 workstation/manage.py setup --apply
```

`workstation/run-windows.cmd` からも同じ引数を使える。`doctor` の資格情報欄は環境変数の存在だけで、認証・モデル提供・残高・制作成功の証明ではない。設定を作る場合は従来の `.env.example` や設定例からPC内の非公開ファイルを作る。

## 完成物を別PC・Deliveryへ渡す

```powershell
py -3.12 workstation/manage.py package C:/completed/clips C:/AutomationData/publish-kit-001 --mode tiktok
```

完成MP4/MOV/MKV/WebM、対応するSRT/ASS、既存カバーだけをコピーする。本文・字幕時刻・元ファイルを変更せず、動画の再エンコードやAI呼び出しはしない。新しい出力先だけを使う。投稿文は既存名または `--title` の内容であり、AIが新しく考えた文ではない。

字幕は動画と同名の `動画名.srt/.ass` を優先する。`captions.srt/.ass` や `subtitles.srt/.ass` は、そのフォルダの動画が1本で候補も1つの場合だけ使う。対応が曖昧な字幕は付けず、結果と `package.json` の `warnings` に動画名・候補名を記録する。必要なら元字幕を保ったまま動画と同名の複製を用意する。出力先はフォルダリンクや `..` を解決した実際のパスで判定し、元フォルダ内へは作らない。

`manifest.json` は相対パスなのでフォルダごと移せる。ClipForge-Deliveryの「素材の一括取り込み」でこのファイルを指定する。Deliveryが対応するMP4だけがその取り込み一覧に入り、MOV等は変換したふりをせずパック内にそのまま保持する。フォルダには `package.json` と「投稿用メモ.txt」も作る。内部のファイルを保ったままフォルダ全体を移す。

## CapCutや制作状態の素材パスを移行する

```powershell
py -3.12 workstation/manage.py relink C:/transfer/draft_content.json C:/transfer/draft_content-moved.json --old-root C:/old-media --new-root D:/AutomationData/media
```

JSON内の明示的なパス項目だけを書き直し、ID・時刻・字幕本文・他フォルダの参照は保持する。元JSONを上書きせず、素材ファイル自体は移動しない。CapCutは終了してバックアップした複製プロジェクトで使い、新PCの素材が揃ってから対応するJSONを置き換える。非公式形式のため、実際にCapCutで開けるかはそのPCの版で確認する。CapCutの設定画面にある実際のドラフトフォルダを使い、別製品の中国版Jianying形式をそのまま流用しない。

## 投稿先へは下書きとして受け渡す

```powershell
py -3.12 workstation/manage.py postiz-draft --integration-id MY_CHANNEL --content "投稿文" --media-id UPLOADED_MEDIA_ID --media-url https://example.test/media.mp4 --output C:/AutomationData/postiz-draft.json
```

Postizへアップロード済みのID/URLを指定する。ローカルの `C:/...mp4` は公開メディアURLにはならない。このコマンドはPostizのCreatePostDtoに対応する `type=draft` JSONを作るだけで、ネットワーク送信・予約・公開はしない。APIへ渡すときは、自分のPostiz接続と各SNSのアプリ/OAuth設定が別途必要。YouTube投稿枠・TikTokの公開権限・X等のAPI条件は別であり、CapCut Proでは代用できない。

## 引き継ぐもの

1. ソースはGitから取得し直す。既存作業は先に保存し、`workspace-lock.json` に記録されたSHAを参照する。
2. 動画・字幕・素材パック・アプリのDBとログ・再開用の状態ファイルを、PCの非公開データフォルダへコピーする。DBはアプリを終了してからコピーする。
3. 状態内の旧絶対パスがある場合は複製JSONへrelinkを使う。元媒体・完了済み動画・親版を保持する。
4. GitHub接続はPCで `gh auth login`、Googleは各アプリの説明に従って既存アカウントでADC等を用意する。ActionsのOIDCトークンはPCへ移せない。サービスアカウントの長期鍵を新規に発行したり、別アカウントへ自動切替したりしない。
5. 既存Actions Runがある場合はそのIDから監視・回収を再開する。PC移行だけを理由に制作や有料APIを再投入しない。

各リポジトリは独立して動作する。他のリポジトリのPython、DB、コードを必須にしていない。統合するのはファイル受け渡しとPC上の配置・導入方法である。

## StreamCut の選定

時刻表を保ったまま、完成MP4/SRT/ASSを移行・納品用にまとめる。ClipForgeの分散制御や別のASRを追加しない。

WhisperXのGPU実行にはCUDA/VRAMが必要。Core i7/16GB/有線LANだけでは満たさないため、既存Colab手順を維持する。今回のsetupは基本パッケージだけでGPUモデルを取得しない。

導入後の既存入口: `.venv\Scripts\streamcut --help`。
