# Battle Studio frontend

[User guide](../../docs/play-gui.md) · [Launcher flags](../../docs/reference/play-gui.md)

Vanilla HTML, CSS and JavaScript, served by `mgba/play_gui.py` and
`src/firered_rl/play_web.py`. Run the Python launcher; opening `index.html`
directly does not provide the team/catalog or generation APIs. No npm build is
needed. Pokémon sprites are included with [source notices](sprites/README.md).

The service binds to loopback and serves only explicit frontend assets and JSON
endpoints. Mutating requests require the page's session token and a local origin.
Browser-provided teams are validated against the Gen III catalog before files
are written or child tools are started. Checkpoint choices refer to server-side
registered paths. One owned job can run at a time.

The browser submits two resolved team specifications. The service invokes the
existing state builder with an isolated output directory, verifies the resulting
state hashes, then invokes the existing human-play runner. Cancellation signals
the child tools so their normal emulator cleanup runs. The UI polls progress and
shows the final report; it does not render an emulator or send game button input.
