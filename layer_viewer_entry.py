"""KiCad IPC action; the launcher supplies the socket and token environment."""
from layer_viewer.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main(plugin=True))
