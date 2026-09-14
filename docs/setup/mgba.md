# Build mGBA

[Dependencies](README.md) · [Create the first save](first-save.md) · [Emulator utilities](../reference/README.md)

Use the project-local mGBA build for scripting and native linked pairs. The
source is pinned to **0.10.5**, commit
`26b7884bc25a5933960f3cdcd98bac1ae14d42e2`.

For a first installation, follow the sections below in order. The
[complete installation page](README.md#3-build-the-patched-mgba-emulator) also
includes the Python prerequisites. The update commands are at the end of this page.

## Build dependencies

```bash
sudo apt update
sudo apt install \
  build-essential cmake git pkg-config xvfb xauth \
  qtbase5-dev qttools5-dev-tools qtmultimedia5-dev libqt5opengl5-dev \
  libsdl2-dev liblua5.4-dev libzip-dev zipcmp zipmerge ziptool zlib1g-dev \
  libedit-dev libminizip-dev libpng-dev \
  libavcodec-dev libavformat-dev libavfilter-dev libswscale-dev
```

## Clone, configure and install

From the project root, with the [Python environment](python.md) active:

```bash
mkdir -p third_party
git clone --branch 0.10.5 --depth 1 \
  https://github.com/mgba-emu/mgba.git third_party/mgba

python tools/setup_mgba_link.py

cmake -S third_party/mgba -B build/mgba \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_INSTALL_PREFIX="$PWD/.local/mgba" \
  -DBUILD_QT=ON \
  -DENABLE_SCRIPTING=ON

cmake --build build/mgba --parallel
cmake --install build/mgba
./.local/mgba/bin/mgba-qt --version
```

Run each command after the preceding one succeeds. `setup_mgba_link.py` applies
the project's autoload, local-link and socket-lifecycle patches and checks the
source commit. It also recognizes patches already applied. See its
[reference](../reference/developer-tools.md#setup-mgba-link).

For an existing pinned checkout, skip `git clone`, rerun the patch helper and
CMake commands. Keep `--parallel` for compilation; use `--parallel 4` to limit
concurrent build jobs when needed.

## Verify the ROM and scripting

Only Pokémon FireRed USA/Europe v1.0 is supported by the current emulator profile:

| Identity | Value |
|---|---|
| Profile | `firered_us_v1_0` |
| Game code | `BPRE` (mGBA may display `AGB-BPRE`) |
| SHA-1 | `41cb23d8dccc8ebd7c649cd8fbb58eeace6e2fdc` |
| CRC32 | `dd88761c` |

```bash
sha1sum FireRed.gba
./.local/mgba/bin/mgba-qt FireRed.gba
```

Confirm **Tools → Scripting…** is available. The launchers load the adapter and
assign ports automatically. Other ROM revisions, LeafGreen, hacks and translations
need a separately validated [ROM profile](../../mgba/profiles/README.md).

## Visible and headless operation

Launchers show windows by default. Add `--headless` to run the same Qt frontend
under `xvfb-run -a`. Save-state generation is the exception: it defaults to
headless and accepts `--visible` for inspection.

All emulator launchers accept `--speed N` or `--speed unlimited`; the default is
normal speed (`1`). There is no fixed maximum useful multiplier: host resources
limit the speed actually achieved. [Benchmark throughput](../benchmark.md) before
choosing a larger environment pool.

Continue with [save-editor setup](save-editor.md) and [the initial save](first-save.md).
For build failures, see [troubleshooting](../troubleshooting.md).

## Updating an existing build

After updating project files, install the current patches, rebuild and install:

```bash
python tools/setup_mgba_link.py
cmake --build build/mgba --parallel
cmake --install build/mgba
```

These commands require the source checkout and configured build from above.
They include runtime speed control and the hidden AI window used by human play.
Restart the launchers after installation so they use the rebuilt executable.
