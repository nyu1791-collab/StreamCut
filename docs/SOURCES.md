# 参考にした実装と採否

2026-10-24に公開文書・実装を調査。以下のコードを丸ごとコピーせず、責務と設計の良い点を参考にして新規実装した。依存ライブラリ自体のライセンスはそれぞれの配布物に従う。

| 一次資料 | 判断 |
|---|---|
| [WhisperX](https://github.com/m-bain/whisperX) / [ASR](https://github.com/m-bain/whisperX/blob/main/whisperx/asr.py) / [alignment](https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py) | ASRとalignmentを分離。全編はバッチ認識、採用区間だけ時刻精密化。日本語の文字単位出力と欠損時刻を扱う |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) / [VAD](https://github.com/SYSTRAN/faster-whisper/blob/v1.2.1/faster_whisper/vad.py) | WhisperXのバックエンド。既存Silero ONNX VADを分割点・発話保護へ使う |
| [ClipForge](https://github.com/nyu1791-collab/ClipForge/tree/39f4edf5681c47bc158e93fba32ea86af634b712) | 日本語字幕の語割れ防止、共通の編集区間表、キャッシュ、出力検査の考え方 |
| [Auto-Editor](https://github.com/WyattBlue/auto-editor) | 余白・短い区間の整理・編集区間表。別アプリとしては追加しない |
| [auto-video-editor](https://github.com/kiy0ni/auto-video-editor/blob/main/auto_video_editor/moments.py) | quiet AND wordlessによるゲーム音の保護。今回はさらにVADの発話区間も保護する |
| [LosslessCut](https://github.com/mifi/lossless-cut/blob/master/docs/troubleshooting.md) | smart cut・任意時刻のcopyの制約を確認。初版は再エンコード |
| [stable-ts](https://github.com/jianfch/stable-ts) | 時刻調整・whisperless版の依存を調査。調査時archivedで、新規の必須依存にしない |
| [ClipsAI](https://github.com/ClipsAI/clipsai) | 意味による話題分割は参考。追加の文埋め込みモデルは増やさない |
| [FunClip](https://github.com/modelscope/FunClip) | 本文を参照して切り抜く考え方。全アプリは取り込まない |
| [live-stream-clipper](https://github.com/UndefBhvr/live-stream-clipper) | ASR→LLM→切り出しの小さな流れ。copyによる切り出しは採らない |
| [livestream-highlight-clipper](https://github.com/Cbhhhh211/livestream-highlight-clipper) | 候補を確認する流れ。Web UI・学習機能は増やさない |
| [autoclip-core](https://github.com/branmendo/autoclip-core) | 小さなCLIとカット予定の保存 |
| [tightcut](https://github.com/AndreaGiulianini/tightcut) | keep区間の表現と一括フィルター。smart encodeは見送る |
| [Silero VAD](https://github.com/snakers4/silero-vad) | 発話検出。モデルを新しく学習しない |
| [yt-dlp](https://github.com/yt-dlp/yt-dlp) | 入力取得。JavaScript実行環境とサイトごとの取得制約は上流に従う |
| [FFmpeg filters](https://ffmpeg.org/ffmpeg-filters.html) | trim/atrim/concat、PTS、ASS、音声フェード |
| [NVIDIA FFmpeg](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.1/ffmpeg-with-nvidia-gpu/index.html) | NVENCとCUDA推論、CPUフィルターは別の処理として扱う |

WhisperXはPyPIの安定版3.8.6、faster-whisperは1.2.1を指定。公開リポジトリのmainにある追加機能を、固定した安定版で利用できると仮定しない。
