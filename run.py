import os
from pathlib import Path

import PyQt5

qt_platforms = Path(PyQt5.__file__).resolve().parent / "Qt5" / "plugins" / "platforms"
if qt_platforms.is_dir():
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(qt_platforms))

from app.main import main


if __name__ == "__main__":
    raise SystemExit(main())
