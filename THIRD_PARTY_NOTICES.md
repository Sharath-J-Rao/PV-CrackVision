# Third-party notices

PV CrackVision bundles or loads the open-source components below. Each component remains under its own license; the full license text of every npm package ships in that package's folder inside the application bundle. This file is generated from `package-lock.json` by `npm run notices` and is verified by `npm run notices:check` (and in CI) so it cannot drift from the locked dependencies.

## Summary by license

| License | Packages |
|---|---|
| MIT | 177 |
| Apache-2.0 | 14 |
| MPL-2.0 | 13 |
| LGPL-3.0-or-later | 10 |
| ISC | 6 |
| Apache-2.0 AND LGPL-3.0-or-later | 3 |
| BSD-3-Clause | 3 |
| 0BSD | 2 |
| (MIT AND Zlib) | 1 |
| (MIT OR CC0-1.0) | 1 |
| (MIT OR GPL-3.0-or-later) | 1 |
| Apache-2.0 AND LGPL-3.0-or-later AND MIT | 1 |
| BSD-2-Clause | 1 |

## Components bundled at packaging time (not in the npm lock)

- **Electron** and **Chromium** (MIT and many third-party licenses). The installer includes Electron's `LICENSE` and `LICENSES.chromium.html`.
- **sharp / libvips** native binaries (Apache-2.0, LGPL-3.0-or-later): distributed as separate dynamically loaded libraries under `app.asar.unpacked`, so the LGPL libraries can be replaced by the user.
- **ONNX Runtime** (MIT).

## Components installed on the user's machine on request (Python training environment)

The optional "Train cell detector" feature sets up a local Python environment and runs `training/train_cell_yolo.py`, which imports **Ultralytics YOLO** (AGPL-3.0, with a separate commercial Enterprise license available from Ultralytics). It is downloaded by the user's machine at training time and is not part of this repository or the installer. **Legal review is required before commercial redistribution or hosted use** (see SECURITY.md, "Commercial-release gates").

## npm production dependencies

| Package | Version | License |
|---|---|---|
| @emnapi/core | 1.11.3 | MIT |
| @emnapi/runtime | 1.11.3 | MIT |
| @emnapi/wasi-threads | 1.2.3 | MIT |
| @esbuild/aix-ppc64 | 0.28.2 | MIT |
| @esbuild/android-arm | 0.28.2 | MIT |
| @esbuild/android-arm64 | 0.28.2 | MIT |
| @esbuild/android-x64 | 0.28.2 | MIT |
| @esbuild/darwin-arm64 | 0.28.2 | MIT |
| @esbuild/darwin-x64 | 0.28.2 | MIT |
| @esbuild/freebsd-arm64 | 0.28.2 | MIT |
| @esbuild/freebsd-x64 | 0.28.2 | MIT |
| @esbuild/linux-arm | 0.28.2 | MIT |
| @esbuild/linux-arm64 | 0.28.2 | MIT |
| @esbuild/linux-ia32 | 0.28.2 | MIT |
| @esbuild/linux-loong64 | 0.28.2 | MIT |
| @esbuild/linux-mips64el | 0.28.2 | MIT |
| @esbuild/linux-ppc64 | 0.28.2 | MIT |
| @esbuild/linux-riscv64 | 0.28.2 | MIT |
| @esbuild/linux-s390x | 0.28.2 | MIT |
| @esbuild/linux-x64 | 0.28.2 | MIT |
| @esbuild/netbsd-arm64 | 0.28.2 | MIT |
| @esbuild/netbsd-x64 | 0.28.2 | MIT |
| @esbuild/openbsd-arm64 | 0.28.2 | MIT |
| @esbuild/openbsd-x64 | 0.28.2 | MIT |
| @esbuild/openharmony-arm64 | 0.28.2 | MIT |
| @esbuild/sunos-x64 | 0.28.2 | MIT |
| @esbuild/win32-arm64 | 0.28.2 | MIT |
| @esbuild/win32-ia32 | 0.28.2 | MIT |
| @esbuild/win32-x64 | 0.28.2 | MIT |
| @img/colour | 1.1.0 | MIT |
| @img/sharp-darwin-arm64 | 0.35.5 | Apache-2.0 |
| @img/sharp-darwin-x64 | 0.35.5 | Apache-2.0 |
| @img/sharp-freebsd-wasm32 | 0.35.5 | Apache-2.0 |
| @img/sharp-libvips-darwin-arm64 | 1.3.4 | LGPL-3.0-or-later |
| @img/sharp-libvips-darwin-x64 | 1.3.4 | LGPL-3.0-or-later |
| @img/sharp-libvips-linux-arm | 1.3.4 | LGPL-3.0-or-later |
| @img/sharp-libvips-linux-arm64 | 1.3.4 | LGPL-3.0-or-later |
| @img/sharp-libvips-linux-ppc64 | 1.3.4 | LGPL-3.0-or-later |
| @img/sharp-libvips-linux-riscv64 | 1.3.4 | LGPL-3.0-or-later |
| @img/sharp-libvips-linux-s390x | 1.3.4 | LGPL-3.0-or-later |
| @img/sharp-libvips-linux-x64 | 1.3.4 | LGPL-3.0-or-later |
| @img/sharp-libvips-linuxmusl-arm64 | 1.3.4 | LGPL-3.0-or-later |
| @img/sharp-libvips-linuxmusl-x64 | 1.3.4 | LGPL-3.0-or-later |
| @img/sharp-linux-arm | 0.35.5 | Apache-2.0 |
| @img/sharp-linux-arm64 | 0.35.5 | Apache-2.0 |
| @img/sharp-linux-ppc64 | 0.35.5 | Apache-2.0 |
| @img/sharp-linux-riscv64 | 0.35.5 | Apache-2.0 |
| @img/sharp-linux-s390x | 0.35.5 | Apache-2.0 |
| @img/sharp-linux-x64 | 0.35.5 | Apache-2.0 |
| @img/sharp-linuxmusl-arm64 | 0.35.5 | Apache-2.0 |
| @img/sharp-linuxmusl-x64 | 0.35.5 | Apache-2.0 |
| @img/sharp-wasm32 | 0.35.5 | Apache-2.0 AND LGPL-3.0-or-later AND MIT |
| @img/sharp-webcontainers-wasm32 | 0.35.5 | Apache-2.0 |
| @img/sharp-win32-arm64 | 0.35.5 | Apache-2.0 AND LGPL-3.0-or-later |
| @img/sharp-win32-ia32 | 0.35.5 | Apache-2.0 AND LGPL-3.0-or-later |
| @img/sharp-win32-x64 | 0.35.5 | Apache-2.0 AND LGPL-3.0-or-later |
| @jridgewell/gen-mapping | 0.3.13 | MIT |
| @jridgewell/remapping | 2.3.5 | MIT |
| @jridgewell/resolve-uri | 3.1.2 | MIT |
| @jridgewell/sourcemap-codec | 1.6.0 | MIT |
| @jridgewell/trace-mapping | 0.3.31 | MIT |
| @napi-rs/wasm-runtime | 1.2.2 | MIT |
| @oxc-project/types | 0.156.0 | MIT |
| @rolldown/binding-android-arm-eabi | 1.2.16 | MIT |
| @rolldown/binding-android-arm64 | 1.2.16 | MIT |
| @rolldown/binding-darwin-arm64 | 1.2.16 | MIT |
| @rolldown/binding-darwin-x64 | 1.2.16 | MIT |
| @rolldown/binding-freebsd-x64 | 1.2.16 | MIT |
| @rolldown/binding-linux-arm-gnueabihf | 1.2.16 | MIT |
| @rolldown/binding-linux-arm64-gnu | 1.2.16 | MIT |
| @rolldown/binding-linux-arm64-musl | 1.2.16 | MIT |
| @rolldown/binding-linux-ppc64-gnu | 1.2.16 | MIT |
| @rolldown/binding-linux-s390x-gnu | 1.2.16 | MIT |
| @rolldown/binding-linux-x64-gnu | 1.2.16 | MIT |
| @rolldown/binding-linux-x64-musl | 1.2.16 | MIT |
| @rolldown/binding-openharmony-arm64 | 1.2.16 | MIT |
| @rolldown/binding-win32-arm64-msvc | 1.2.16 | MIT |
| @rolldown/binding-win32-x64-msvc | 1.2.16 | MIT |
| @rolldown/pluginutils | 1.0.1 | MIT |
| @tailwindcss/node | 4.3.4 | MIT |
| @tailwindcss/oxide | 4.3.4 | MIT |
| @tailwindcss/oxide-android-arm64 | 4.3.4 | MIT |
| @tailwindcss/oxide-darwin-arm64 | 4.3.4 | MIT |
| @tailwindcss/oxide-darwin-x64 | 4.3.4 | MIT |
| @tailwindcss/oxide-freebsd-x64 | 4.3.4 | MIT |
| @tailwindcss/oxide-linux-arm-gnueabihf | 4.3.4 | MIT |
| @tailwindcss/oxide-linux-arm64-gnu | 4.3.4 | MIT |
| @tailwindcss/oxide-linux-arm64-musl | 4.3.4 | MIT |
| @tailwindcss/oxide-linux-x64-gnu | 4.3.4 | MIT |
| @tailwindcss/oxide-linux-x64-musl | 4.3.4 | MIT |
| @tailwindcss/oxide-wasm32-wasi | 4.3.4 | MIT |
| @tailwindcss/oxide-win32-arm64-msvc | 4.3.4 | MIT |
| @tailwindcss/oxide-win32-x64-msvc | 4.3.4 | MIT |
| @tailwindcss/vite | 4.3.4 | MIT |
| @tybys/wasm-util | 0.10.3 | MIT |
| @types/node | 22.20.5 | MIT |
| @vitejs/plugin-react | 6.2.1 | MIT |
| @xmldom/xmldom | 0.9.12 | MIT |
| accepts | 1.3.8 | MIT |
| adm-zip | 0.6.1 | MIT |
| array-flatten | 1.1.1 | MIT |
| body-parser | 1.20.9 | MIT |
| bytes | 3.1.2 | MIT |
| call-bind-apply-helpers | 1.0.2 | MIT |
| call-bound | 1.0.4 | MIT |
| content-disposition | 0.5.4 | MIT |
| content-type | 1.0.5 | MIT |
| cookie | 0.7.2 | MIT |
| cookie-signature | 1.0.7 | MIT |
| core-util-is | 1.0.3 | MIT |
| debug | 2.6.9 | MIT |
| define-data-property | 1.1.4 | MIT |
| define-properties | 1.2.1 | MIT |
| depd | 2.0.0 | MIT |
| destroy | 1.3.0 | MIT |
| detect-libc | 2.1.2 | Apache-2.0 |
| dotenv | 17.4.2 | BSD-2-Clause |
| dunder-proto | 1.0.1 | MIT |
| ee-first | 1.1.1 | MIT |
| encodeurl | 2.0.1 | MIT |
| enhanced-resolve | 5.26.1 | MIT |
| es-define-property | 1.0.1 | MIT |
| es-errors | 1.3.0 | MIT |
| es-object-atoms | 1.1.4 | MIT |
| esbuild | 0.28.2 | MIT |
| escape-html | 1.0.3 | MIT |
| escape-string-regexp | 4.0.0 | MIT |
| etag | 1.8.1 | MIT |
| exifreader | 4.54.0 | MPL-2.0 |
| express | 4.22.3 | MIT |
| fdir | 6.5.0 | MIT |
| finalhandler | 1.3.2 | MIT |
| forwarded | 0.2.0 | MIT |
| fresh | 0.5.2 | MIT |
| fsevents | 2.3.3 | MIT |
| function-bind | 1.1.4 | MIT |
| get-intrinsic | 1.3.0 | MIT |
| get-proto | 1.0.1 | MIT |
| global-agent | 4.1.7 | BSD-3-Clause |
| globalthis | 1.0.4 | MIT |
| gopd | 1.3.0 | MIT |
| graceful-fs | 4.2.11 | ISC |
| has-property-descriptors | 1.0.2 | MIT |
| has-symbols | 1.1.0 | MIT |
| hasown | 2.0.4 | MIT |
| http-errors | 2.0.1 | MIT |
| iconv-lite | 0.4.24 | MIT |
| immediate | 3.0.6 | MIT |
| inherits | 2.0.4 | ISC |
| ipaddr.js | 1.9.1 | MIT |
| isarray | 1.0.0 | MIT |
| jiti | 2.7.0 | MIT |
| jszip | 3.10.2 | (MIT OR GPL-3.0-or-later) |
| lie | 3.3.0 | MIT |
| lightningcss | 1.33.0 | MPL-2.0 |
| lightningcss-android-arm64 | 1.33.0 | MPL-2.0 |
| lightningcss-darwin-arm64 | 1.33.0 | MPL-2.0 |
| lightningcss-darwin-x64 | 1.33.0 | MPL-2.0 |
| lightningcss-freebsd-x64 | 1.33.0 | MPL-2.0 |
| lightningcss-linux-arm-gnueabihf | 1.33.0 | MPL-2.0 |
| lightningcss-linux-arm64-gnu | 1.33.0 | MPL-2.0 |
| lightningcss-linux-arm64-musl | 1.33.0 | MPL-2.0 |
| lightningcss-linux-x64-gnu | 1.33.0 | MPL-2.0 |
| lightningcss-linux-x64-musl | 1.33.0 | MPL-2.0 |
| lightningcss-win32-arm64-msvc | 1.33.0 | MPL-2.0 |
| lightningcss-win32-x64-msvc | 1.33.0 | MPL-2.0 |
| lucide-react | 0.546.0 | ISC |
| magic-string | 1.4.5 | MIT |
| matcher | 4.0.0 | MIT |
| math-intrinsics | 1.1.0 | MIT |
| media-typer | 0.3.0 | MIT |
| merge-descriptors | 1.0.3 | MIT |
| methods | 1.1.4 | MIT |
| mime | 1.6.0 | MIT |
| mime-db | 1.52.0 | MIT |
| mime-types | 2.1.35 | MIT |
| ms | 2.0.0 | MIT |
| ms | 2.1.3 | MIT |
| nanoid | 3.3.20 | MIT |
| negotiator | 0.6.3 | MIT |
| object-inspect | 1.13.4 | MIT |
| object-keys | 1.1.1 | MIT |
| on-finished | 2.4.2 | MIT |
| onnxruntime-common | 1.31.1 | MIT |
| onnxruntime-node | 1.31.1 | MIT |
| pako | 1.0.11 | (MIT AND Zlib) |
| parseurl | 1.3.3 | MIT |
| path-to-regexp | 0.1.13 | MIT |
| picocolors | 1.1.1 | ISC |
| picomatch | 4.0.7 | MIT |
| postcss | 8.5.32 | MIT |
| process-nextick-args | 2.0.1 | MIT |
| proxy-addr | 2.0.8 | MIT |
| qs | 6.16.2 | BSD-3-Clause |
| range-parser | 1.2.1 | MIT |
| raw-body | 2.5.3 | MIT |
| react | 19.3.1 | MIT |
| react-dom | 19.3.1 | MIT |
| readable-stream | 2.3.8 | MIT |
| rolldown | 1.2.16 | MIT |
| safe-buffer | 5.1.2 | MIT |
| safe-buffer | 5.2.1 | MIT |
| safer-buffer | 2.1.2 | MIT |
| scheduler | 0.28.0 | MIT |
| semver | 7.8.6 | ISC |
| send | 0.19.2 | MIT |
| serialize-error | 8.1.0 | MIT |
| serve-static | 1.16.3 | MIT |
| setimmediate | 1.0.5 | MIT |
| setprototypeof | 1.3.0 | ISC |
| sharp | 0.35.5 | Apache-2.0 |
| side-channel | 1.1.1 | MIT |
| side-channel-list | 1.0.1 | MIT |
| side-channel-map | 1.0.1 | MIT |
| side-channel-weakmap | 1.0.2 | MIT |
| source-map-js | 1.2.2 | BSD-3-Clause |
| statuses | 2.0.2 | MIT |
| string_decoder | 1.1.1 | MIT |
| tailwindcss | 4.3.4 | MIT |
| tapable | 2.3.3 | MIT |
| tinyglobby | 0.2.17 | MIT |
| toidentifier | 1.0.1 | MIT |
| tslib | 2.8.1 | 0BSD |
| tslib | 2.9.1 | 0BSD |
| tsx | 4.23.15 | MIT |
| type-fest | 0.20.2 | (MIT OR CC0-1.0) |
| type-is | 1.6.18 | MIT |
| undici-types | 6.21.0 | MIT |
| unpipe | 1.0.0 | MIT |
| util-deprecate | 1.0.2 | MIT |
| utils-merge | 1.0.1 | MIT |
| vary | 1.1.4 | MIT |
| vite | 8.3.5 | MIT |
