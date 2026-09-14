# Showdown setup

[Quickstart](../quickstart.md) · [Dependencies](README.md) · [Simulator training](../showdown-training.md)

Complete [installation step 1](README.md#1-install-python-and-the-training-packages)
for Python, then [installation step 2](README.md#2-install-nodejs-and-the-local-showdown-engine)
for the Node.js 24 installation commands. A Node installation includes npm.
The package minimum remains Node.js 18; new setups use an LTS release.

## Install the pinned simulator

With Node available, run from the project root:

```bash
node --version
npm --version
npm --prefix sim/showdown ci --omit=optional
node -e "const {Dex} = require('./sim/showdown/node_modules/pokemon-showdown'); console.log(Dex.mod('gen3').species.get('Charizard').name)"
```

The last command prints `Charizard`. `npm ci` uses [the lockfile](../../sim/showdown/package-lock.json)
to install Pokémon Showdown 0.11.11 under `sim/showdown/node_modules/`.
`--omit=optional` leaves out optional server/database packages. The adapter uses
a private simulator API, so retain the pinned version.

## In a new terminal

Return to the project root, activate Python, and select the installed Node version:

```bash
source .venv/bin/activate
export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
source "$NVM_DIR/nvm.sh"
nvm use 24
```

The nvm lines apply when you used the installation guide's nvm route. A working
system Node installation does not need them. Install npm dependencies once per
checkout, or again after an update changes the lockfile.

Training runs locally, headless and unthrottled. It reads JSON teams directly;
it does not require a ROM, save files, .NET, Xvfb, a browser or a Showdown account.
Dependency installation needs a network connection; battle simulation does not.

If npm reports deprecation or audit warnings, see [dependency warning guidance](../troubleshooting.md#npm-installs-with-dependency-warnings)
before changing the lockfile. Continue with [the quickstart training command](../quickstart.md#4-train-an-agent).
