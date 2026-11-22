"""Execute a .ipynb top to bottom with jupyter_client and store outputs in place.

usage: python run_notebook.py NOTEBOOK [--timeout SECONDS]
"""
import json
import queue
import sys
import time

from jupyter_client.manager import start_new_kernel


def run(path, timeout=7200, cwd=None):
    nb = json.load(open(path))
    km, kc = start_new_kernel(kernel_name="python3", cwd=cwd)
    failed = None
    try:
        count = 0
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            count += 1
            t0 = time.time()
            msg_id = kc.execute(src)
            outputs = []
            while True:
                try:
                    msg = kc.get_iopub_msg(timeout=timeout)
                except queue.Empty:
                    raise TimeoutError(f"cell {i} timed out")
                if msg["parent_header"].get("msg_id") != msg_id:
                    continue
                t, c = msg["msg_type"], msg["content"]
                if t == "status" and c["execution_state"] == "idle":
                    break
                if t == "stream":
                    if outputs and outputs[-1]["output_type"] == "stream" and outputs[-1]["name"] == c["name"]:
                        outputs[-1]["text"] += c["text"]
                    else:
                        outputs.append({"output_type": "stream", "name": c["name"], "text": c["text"]})
                elif t in ("execute_result", "display_data"):
                    o = {"output_type": t, "data": c["data"], "metadata": c.get("metadata", {})}
                    if t == "execute_result":
                        o["execution_count"] = count
                    outputs.append(o)
                elif t == "error":
                    outputs.append({"output_type": "error", "ename": c["ename"], "evalue": c["evalue"],
                                    "traceback": c["traceback"]})
                    failed = (i, c["ename"], c["evalue"])
            for o in outputs:
                if o["output_type"] == "stream":
                    o["text"] = o["text"].splitlines(keepends=True)
            cell["outputs"] = outputs
            cell["execution_count"] = count
            print(f"cell {i:3d} done in {time.time() - t0:7.1f}s", flush=True)
            json.dump(nb, open(path, "w"), indent=1)
            if failed:
                print("FAILED", failed, flush=True)
                print("\n".join(outputs[-1].get("traceback", []))[-4000:])
                break
    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)
    return failed


if __name__ == "__main__":
    import os
    nb_path = sys.argv[1]
    result = run(nb_path, cwd=sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(nb_path)))
    sys.exit(1 if result else 0)
