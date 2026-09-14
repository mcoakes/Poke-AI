# Install dependencies

[Quickstart](../quickstart.md) · [Documentation](../README.md) · [Project overview](../../README.md)

These commands target **Linux Mint 22.1 / Ubuntu 24.04**, using Bash. Open a
terminal in the extracted project directory: `README.md`, `pyproject.toml`,
`mgba/` and `config/` should be here. Run each command after the previous one
succeeds. No administrator privileges are needed for the Python or npm installs.

Choose what you want to run:

- **The complete quickstart:** complete steps 1–4 below. Human play also needs a
  desktop session, a browser, your FireRed ROM and the initial save from step 2
  of the quickstart.
- **Showdown training and campaigns only:** complete steps 1–2. Team JSONs are
  included; you can begin without a ROM or save file.
- **mGBA training or human play only:** complete steps 1, 3 and 4. Node.js is
  only used by the Showdown backend.
- **Generate team JSONs or run mock examples:** step 1 is enough.

The sections below contain the full installation sequence. The linked component
pages explain alternatives and maintenance.

## 1. Install Python and the training packages

Python 3.11 or newer is required; Ubuntu 24.04/Mint 22.1 provide Python 3.12.
The training extras also provide the model loader used during human play.

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip git curl ca-certificates unzip xz-utils
python3 --version

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[train]'
```

Check the environment:

```bash
python -c "import gymnasium, numpy, torch, sb3_contrib, firered_rl; print('Python training dependencies ready')"
python examples/random_agent.py
```

The first command should print `Python training dependencies ready`. The second
runs a mock battle and prints an episode reward; its exact value can vary.

In **each new terminal**, return to the project directory and activate Python:

```bash
source .venv/bin/activate
```

For optional development packages and tests, see [Python options](python.md).

## 2. Install Node.js and the local Showdown engine

For a new installation, use **Node.js 24 LTS**, which includes npm. The project's
minimum is Node.js 18; the setup below selects a supported LTS release.
[Node.js release schedule](https://nodejs.org/en/about/previous-releases).

If `node --version` already reports 24.x, skip the nvm block. Otherwise install
Node through [nvm](https://github.com/nvm-sh/nvm#installing-and-updating):

```bash
export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.7/install.sh \
  -o /tmp/firered-nvm-install.sh
bash /tmp/firered-nvm-install.sh
source "$NVM_DIR/nvm.sh"
nvm install 24
nvm use 24
```

Then install the pinned simulator, from the project root:

```bash
node --version
npm --version
npm --prefix sim/showdown ci --omit=optional
node -e "const {Dex} = require('./sim/showdown/node_modules/pokemon-showdown'); console.log(Dex.mod('gen3').species.get('Charizard').name)"
```

Expected: a Node version starting with `v24.`, an npm version, then `Charizard`.
Pokémon Showdown is installed locally at
`sim/showdown/node_modules/pokemon-showdown/`; training does not contact a public
Showdown server. Keep the lockfile when installing updates. See
[Showdown setup](showdown.md) for dependency warnings and terminal setup.

**Showdown-only setup is complete.** Continue at [quickstart step 3](../quickstart.md#3-build-and-customize-teams).

## 3. Build the patched mGBA emulator

Install compiler, Qt/Lua and virtual-display dependencies:

```bash
sudo apt install \
  build-essential cmake pkg-config xvfb xauth \
  qtbase5-dev qttools5-dev-tools qtmultimedia5-dev libqt5opengl5-dev \
  libsdl2-dev liblua5.4-dev libzip-dev zipcmp zipmerge ziptool zlib1g-dev \
  libedit-dev libminizip-dev libpng-dev \
  libavcodec-dev libavformat-dev libavfilter-dev libswscale-dev
```

Fetch the pinned source. **Only run the clone on the first installation**; if
`third_party/mgba` already exists, continue with the patch helper.

```bash
mkdir -p third_party
git clone --branch 0.10.5 --depth 1 \
  https://github.com/mgba-emu/mgba.git third_party/mgba
```

Apply the project's scripting, link-session and runtime-control patches, then
configure, compile and install:

```bash
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

Expected version: **0.10.5**, commit
`26b7884bc25a5933960f3cdcd98bac1ae14d42e2`. The patch helper checks that commit.
The executable is `.local/mgba/bin/mgba-qt`; the launchers use it by default.
If compiling exhausts memory, rerun the build with `--parallel 2`.

Open the emulator and confirm **Tools → Scripting…** exists:

```bash
./.local/mgba/bin/mgba-qt
```

Close it before continuing. No script needs to be loaded manually: the project
launchers do that. `xvfb-run` provides headless execution of this same Qt build.
After future patch updates, follow the [rebuild instructions](mgba.md#updating-an-existing-build).

## 4. Install the .NET 10 SDK and save editor

The SDK compiles the PKHeX.Core-based editor. A runtime-only installation is
insufficient. This sequence installs the SDK inside the project so it also works
when an apt source does not provide .NET 10.

Install its Ubuntu 24.04 system libraries:

```bash
sudo apt install \
  ca-certificates libc6 libgcc-s1 libgssapi-krb5-2 libicu74 \
  libssl3t64 libstdc++6 tzdata zlib1g
```

Install and select the SDK:

```bash
curl -fsSL https://dot.net/v1/dotnet-install.sh -o /tmp/firered-dotnet-install.sh
bash /tmp/firered-dotnet-install.sh \
  --channel 10.0 --quality GA --install-dir "$PWD/.local/dotnet"
export DOTNET_ROOT="$PWD/.local/dotnet"
export PATH="$DOTNET_ROOT:$PATH"
dotnet --version
```

Expected: `10.0.xxx` (the patch version can differ). These commands follow
Microsoft's [installation script](https://learn.microsoft.com/en-us/dotnet/core/tools/dotnet-install-script)
and [Ubuntu dependencies](https://learn.microsoft.com/en-us/dotnet/core/install/linux-ubuntu-install#ubuntu-2404).
The SDK download does not install those system libraries for you.

Build the save editor:

```bash
bash tools/setup_pkhex.sh
```

Expected: `Build succeeded` and
`tools/pkhex_team_editor/bin/Debug/net10.0/PKHeXTeamEditor.dll`. The helper fetches
PKHeX commit `77dcd3a7895bceaafbbff12d25bdf77c1acd8ca5` automatically.

In a new terminal, use these exports before running `dotnet` or the setup helper:

```bash
export DOTNET_ROOT="$PWD/.local/dotnet"
export PATH="$DOTNET_ROOT:$PATH"
```

The Python save utilities also find the project-local SDK directly. If you
already have a working .NET 10 SDK, see the [existing-SDK option](save-editor.md).

## Continue with your first save

Installation provides the tools; the next steps supply the game data:

1. [Create the initial FireRed battery save](../quickstart.md#2-set-up-the-initial-save-file).
2. [Build the included teams and counter states](../quickstart.md#3-build-and-customize-teams).
3. [Train an agent and play against it](../quickstart.md#4-train-an-agent).

A GPU is optional. Both battle engines run on the CPU; `--device cuda` changes
only the policy's PyTorch device. CPU is the default for this walkthrough.
