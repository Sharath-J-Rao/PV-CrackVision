# v1.4.8.8 Windows launcher validation

- Added `PV-CrackVision.exe`, a native x64 Windows console launcher.
- The launcher resolves its own directory, requires `start.bat` beside itself, and launches that script with the application directory as the working directory.
- PE32+ x86-64 structure and imported Kernel32 entry points were inspected after compilation.
- The executable was cross-compiled with Clang/LLD in the Linux build environment. It was not executed on Windows in this environment; Windows runtime behaviour still needs a smoke test on the target machine.
- This is a launcher, not an all-in-one offline installer. The existing `start.bat` still provisions Node when needed and installs npm dependencies on first run. Internet access is required for first-time setup.
- No user datasets, annotations, trained checkpoints, `.env.local`, Python virtual environments, or `node_modules` are included in the release archive.
