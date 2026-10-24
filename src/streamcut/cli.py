"""Small command surface: run, worker, doctor."""

import argparse
import json
import sys
from pathlib import Path

from . import __version__, media
from .model import Config, CutError
from .pipeline import reexport, run
from .store import read
from .worker import work


def parser():
    p = argparse.ArgumentParser(
        prog="streamcut", description="配信から話題別の字幕付き下書きを生成"
    )
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="取得・解析・選定・仕上げを実行")
    r.add_argument("source", help="ローカル動画または配信アーカイブのHTTPS URL")
    r.add_argument("--project", required=True, type=Path)
    r.add_argument("--config", type=Path)
    r.add_argument("--model", help="LLMモデルID。暗黙の既定値なし")
    r.add_argument("--gpu-workers", type=int)
    r.add_argument("--render-workers", type=int)
    r.add_argument("--clips", type=int)
    r.add_argument("--distributed", action="store_true", help="共有フォルダ上のColabワーカーを使用")
    r.add_argument("--resume", action="store_true")
    r.add_argument("--transcript", type=Path, help="保存済みunit/word時刻を取り込む")
    r.add_argument("--selection", type=Path, help="LLMを呼ばずID指定の選定を取り込む")
    w = sub.add_parser("worker", help="共有フォルダのタスクを処理")
    w.add_argument("project", type=Path)
    w.add_argument("--role", choices=("gpu", "render"), required=True)
    w.add_argument("--rank", type=int, required=True)
    w.add_argument("--timeout", type=int, default=14400)
    e = sub.add_parser("export", help="保存済みalignmentから字幕・見出しを再書き出し")
    e.add_argument("project", type=Path)
    e.add_argument("--config", type=Path, help="字幕・無音・描画設定だけのJSON")
    e.add_argument("--headlines", type=Path, help="01等の番号と新見出しのJSON")
    sub.add_parser("doctor", help="モデルを取得せず実行環境を確認")
    return p


def doctor():
    media.tools()
    from .captions import font_path

    path, index = font_path("Noto Sans CJK JP")
    from .captions import ensure_glyphs

    ensure_glyphs(path, index, "日本語の字幕確認")
    import importlib.util

    try:
        import torch

        gpu_count = torch.cuda.device_count()
    except ImportError:
        gpu_count = 0
    result = {
        "cpu_threads": media.available_cpus(),
        "cuda_devices": gpu_count,
        "whisperx_installed": importlib.util.find_spec("whisperx") is not None,
        "nvenc_usable": media.nvenc_available(),
        "japanese_font": True,
        "asr_executed": False,
        "colab_allocation_checked": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "doctor":
            doctor()
        elif args.command == "worker":
            print(work(args.project, args.role, args.rank, args.timeout))
        elif args.command == "export":
            reexport(
                args.project,
                read(args.config) if args.config else None,
                read(args.headlines) if args.headlines else None,
            )
        else:
            values = read(args.config) if args.config else {}
            for option, field in (
                ("model", "llm_model"),
                ("gpu_workers", "gpu_workers"),
                ("render_workers", "render_workers"),
                ("clips", "clips"),
            ):
                if getattr(args, option) is not None:
                    values[field] = getattr(args, option)
            config = Config.load(values)
            if not args.distributed and (config.gpu_workers != 1 or config.render_workers != 1):
                raise CutError("multiple Colab workers require --distributed")
            run(
                args.source,
                args.project,
                config,
                distributed=args.distributed,
                resume=args.resume,
                transcript=args.transcript,
                selection=args.selection,
            )
    except Exception as error:
        # Only our own error class carries a public message; parser/provider payloads stay private.
        message = str(error) if isinstance(error, CutError) else type(error).__name__
        print("停止: " + message, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("中断しました。同じ設定で --resume できます。", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
