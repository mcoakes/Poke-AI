"""Coordinate two local frame-driven lobby controllers and verify their cable."""
from concurrent.futures import ThreadPoolExecutor, wait
import json
import time

from firered_rl.backends.mgba_bridge import MGBABridgeBackend
from firered_rl.link_test import connection_errors


def connect_pair(ports, session_id, log_path, timeout=180, cancel=None, announce=print,
                 process_state=None, start_battle=False):
    # Quiet callers (including PPO) disable console output with None.
    # Structured connection diagnostics still go to log_path.
    if announce is None:
        announce = lambda message: None
    bridges = [MGBABridgeBackend("127.0.0.1", port) for port in ports]
    started = [False, False]
    success = False
    last = [None, None]
    with ThreadPoolExecutor(max_workers=2) as executor, log_path.open("w") as log:
        def record(**event):
            log.write(json.dumps(event)+"\n")
            log.flush()

        def request(method, indices=(0, 1), **kwargs):
            futures = [executor.submit(getattr(bridges[i], method), **kwargs) for i in indices]
            results, errors = [], []
            try:
                for i, future in zip(indices, futures):
                    try:
                        results.append(future.result())
                    except Exception as error:
                        results.append(None)
                        errors.append(dict(port=ports[i], error=str(error), type=type(error).__name__))
            except BaseException:
                wait(futures)
                raise
            if errors:
                record(event="request_failed", command=method, ports=[ports[i] for i in indices],
                       failures=errors, partial_responses=results, last_reports=last)
                first = errors[0]
                raise RuntimeError(f"port {first['port']} {method}: {first['error']}; see {log_path}")
            return results

        def pause():
            if cancel is not None:
                cancel.wait(0.1)
            else:
                time.sleep(0.1)

        def check_deadline():
            if cancel is not None and cancel.is_set():
                raise RuntimeError("pool connection cancelled")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"auto-connect exceeded {timeout:g}s; last={last}; see {log_path}")

        def progress(indices=(0, 1)):
            states = request("link_connect_status", indices)
            for i, state in zip(indices, states):
                previous = last[i]
                last[i] = state
                if not previous or previous["phase"] != state["phase"]:
                    announce(f"port {ports[i]}: {state['phase']}")
            record(event="progress", reports=last)
            for i, state in zip(indices, states):
                if state.get("error") or state["phase"] in ("cancelled", "client_disconnected", "shutdown"):
                    raise RuntimeError(f"port {ports[i]}: {state.get('error', state['phase'])}; see {log_path}")
            return states

        try:
            statuses = request("status")
            if any(not s.get("auto_connect_supported") or not s.get("staged_connect_supported") for s in statuses):
                raise RuntimeError("auto-connect needs adapter 0.9.2 and its lobby files; restart the launcher")
            if start_battle and any(not s.get("battle_start_supported") for s in statuses):
                raise RuntimeError("battle startup needs adapter 0.10.0 and its profile files; restart the launcher")
            if any(not s.get("native_link_session") or s.get("link_session_id") != session_id for s in statuses):
                raise RuntimeError("adapters are not in the assigned native cable session")
            if {s.get("link_player") for s in statuses} != {1, 2}:
                raise RuntimeError("expected two distinct session players")
            record(event="start", ports=list(ports), session_id=session_id, statuses=statuses,
                   start_battle=start_battle)
            deadline = time.monotonic()+timeout
            # Keep the first player at the actual cable service menu while the
            # second performs its adapter probe. Neither core is suspended.
            for i in range(2):
                check_deadline()
                started[i] = True
                request("link_connect", (i,), hold_at_service=True)
                while True:
                    check_deadline()
                    state = progress((i,))[0]
                    if state.get("probe_complete") and state["phase"] == "waiting_for_partner_probe":
                        break
                    pause()
            record(event="probes_complete", reports=last)
            request("link_connect_continue")
            stable = 0
            while True:
                check_deadline()
                states = progress()
                if all(s["phase"] == "complete" for s in states):
                    reports = request("link_status")
                    errors = connection_errors(reports)
                    record(event="verify", reports=reports, pending=errors)
                    stable = stable+1 if not errors else 0
                    if stable >= 3:
                        result = dict(session_id=session_id, ports=list(ports), reports=reports,
                                      status="connected", location="colosseum", log=str(log_path))
                        record(event="connected", **result)
                        if not start_battle:
                            success = True
                            return result
                        break
                else:
                    stable = 0
                pause()
            announce(f"ports {ports[0]}/{ports[1]}: cable verified; starting battle")
            deadline = time.monotonic()+timeout
            request("link_battle_start")
            stable = 0
            while True:
                check_deadline()
                states = progress()
                if all(s["phase"] == "battle_ready" and not s.get("active")
                       and s.get("diagnostics", {}).get("command_ready") for s in states):
                    reports = request("link_status")
                    errors = connection_errors(reports, require_battle=True)
                    record(event="verify_battle", reports=reports, pending=errors)
                    stable = stable+1 if not errors else 0
                    if stable >= 3:
                        result.update(status="battle_ready", location="battle", reports=reports,
                                      decisions=[s["diagnostics"] for s in states])
                        record(event="battle_ready", **result)
                        success = True
                        return result
                else:
                    stable = 0
                pause()
        except BaseException as error:
            native = None
            if process_state is not None:
                try:
                    native = process_state()
                    announce("Process state before cleanup: "+json.dumps(native))
                except Exception as diagnostic_error:
                    native = {"diagnostic_error": str(diagnostic_error)}
            record(event="failed", error=str(error), last_reports=last, process=native)
            raise
        finally:
            for i, bridge in enumerate(bridges):
                if started[i] and not success:
                    try:
                        bridge.link_connect_cancel()
                    except (OSError, RuntimeError, ValueError):
                        pass
                bridge.close()
