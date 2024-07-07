import sys
import enum

class Os(enum.Enum):
    Windows = enum.auto()
    MacOS   = enum.auto()
    Linux   = enum.auto()

if sys.platform.startswith("win"):
    os = Os.Windows
elif sys.platform.startswith("linux"):
    os = Os.Linux
elif sys.platform.startswith("darwin"):
    os = Os.MacOS
else:
    print("Your system is not officially supported at the moment!\n"
          "You can let me know on GitHub, or you can try porting yourself")
    sys.exit(1)