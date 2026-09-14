#!/usr/bin/env python3
"""Apply pinned autoload, local-link and socket fixes without removing other edits."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party/mgba"
PIN = "26b7884bc25a5933960f3cdcd98bac1ae14d42e2"


def git(*args, check=True):
    return subprocess.run(["git", "-C", str(SOURCE), *map(str, args)],
                          check=check, capture_output=True, text=True)


def main():
    try:
        if git("rev-parse", "HEAD").stdout.strip() != PIN:
            raise RuntimeError("expected the pinned mGBA 0.10.5 commit; source was not changed")
        autoload = ROOT / "patches/mgba-0.10.5-script-autoload.patch"
        link = ROOT / "patches/mgba-0.10.5-link-session.patch"
        if git("apply", "--reverse", "--check", link, check=False).returncode == 0:
            print("Link patch already applied.")
        else:
            if git("apply", "--check", link, check=False).returncode != 0:
                if git("apply", "--check", autoload, check=False).returncode != 0:
                    raise RuntimeError("patches do not match this working tree; no existing edits were removed")
                git("apply", autoload)
            result = git("apply", "--check", link, check=False)
            if result.returncode:
                raise RuntimeError(result.stderr)
            git("apply", link)
            print("Link-session patch applied.")
        lifecycle = ROOT / "patches/mgba-0.10.5-socket-lifecycle.patch"
        if git("apply", "--reverse", "--check", lifecycle, check=False).returncode == 0:
            print("Socket lifecycle fix already applied.")
        else:
            result = git("apply", "--check", lifecycle, check=False)
            if result.returncode:
                raise RuntimeError("socket patch does not match this tree; existing edits were retained:\n" + result.stderr)
            git("apply", lifecycle)
            print("Socket lifecycle fix applied.")
        # Upgrade either previously shipped speed setter (including the QVariant hotfix).
        window = SOURCE / "src/platform/qt/Window.cpp"
        before = window.read_text()
        after = before
        after = after.replace('m_config->setOption("fpsTarget", (16777216.0 / 280896.0) * (ratio == 0 ? 1 : ratio));', 'mCoreConfigSetOverrideFloatValue(m_config->config(), "fpsTarget", (16777216.0 / 280896.0) * (ratio == 0 ? 1 : ratio));')
        after = after.replace('m_config->setOption("fpsTarget", QVariant((16777216.0 / 280896.0) * (ratio == 0 ? 1 : ratio)));', 'mCoreConfigSetOverrideFloatValue(m_config->config(), "fpsTarget", (16777216.0 / 280896.0) * (ratio == 0 ? 1 : ratio));')
        if after != before:
            after = after.replace('m_config->setOption("audioSync", ratio == 0 ? 0 : 1);', 'mCoreConfigSetOverrideIntValue(m_config->config(), "audioSync", ratio == 0 ? 0 : 1);')
            after = after.replace('m_config->setOption("videoSync", 0);', 'mCoreConfigSetOverrideIntValue(m_config->config(), "videoSync", 0);')
            window.write_text(after)
            print("Upgraded runtime speed override setters.")
        speed = ROOT / "patches/mgba-0.10.5-runtime-speed.patch"
        if git("apply", "--reverse", "--check", speed, check=False).returncode == 0:
            print("Runtime speed patch already applied.")
        else:
            result = git("apply", "--check", speed, check=False)
            if result.returncode:
                raise RuntimeError("runtime speed patch does not match this tree; existing edits were retained:\n" + result.stderr)
            git("apply", speed)
            print("Runtime speed patch applied.")
        human_window = ROOT / "patches/mgba-0.10.5-human-window.patch"
        if git("apply", "--reverse", "--check", human_window, check=False).returncode == 0:
            print("Human window patch already applied.")
        else:
            result = git("apply", "--check", human_window, check=False)
            if result.returncode:
                raise RuntimeError("human window patch does not match this tree; existing edits were retained:\n" + result.stderr)
            git("apply", human_window)
            print("Human window patch applied.")
        print("Next: cmake --build build/mgba --parallel")
        print("Then: cmake --install build/mgba")
    except (subprocess.CalledProcessError, RuntimeError) as error:
        raise SystemExit(getattr(error, "stderr", None) or str(error))


if __name__ == "__main__":
    main()
