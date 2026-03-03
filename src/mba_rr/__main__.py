"""Allow `python -m mba_rr` to dispatch to the CLI entrypoint."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
