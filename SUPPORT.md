# PV CrackVision — Support

## Before you contact support
1. Note the **version** (it is part of the installer name, e.g. `PV-CrackVision-<version>-x64.exe`, and the first heading of `CHANGELOG.md`).
2. Note what you did, what you expected and what happened. A screenshot of the on-screen message helps.
3. Attach the desktop log: `%APPDATA%\PV CrackVision\logs\desktop.log` (start-up, crash, hang and unexpected-error records are written here).
4. For training problems also attach the training worker log, found below the data folder (`training/` and `runtime/`; the data folder is the one chosen under **Data & Models** in Settings).

## Privacy
Do **not** send image sets, annotations or the data folder unless support asks and you are permitted to share them: they can contain site imagery and GPS positions. The application works fully offline and sends no usage data or telemetry; nothing is uploaded unless you attach it yourself.

## Before upgrading
Back up the complete data folder. The undo journal is not a backup. Existing labels, annotations, reviews and trained models are read unchanged by newer 1.4.x releases unless the CHANGELOG states otherwise.

## Checking a download
Release builds ship with `SHA256SUMS.txt`. In PowerShell: `Get-FileHash .\PV-CrackVision-<version>-x64.exe -Algorithm SHA256` and compare with the matching line.

## Contact
Support contact and response times: *to be set by the vendor before distribution.*
