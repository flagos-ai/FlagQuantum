"""Check that a registered notebook kernel can execute the installed framework."""

import argparse

from jupyter_client import KernelManager


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", default="python3")
    parser.add_argument("--qsteed", action="store_true")
    args = parser.parse_args()
    manager = KernelManager(kernel_name=args.kernel)
    manager.start_kernel()
    client = manager.blocking_client()
    try:
        client.start_channels()
        client.wait_for_ready(timeout=60)
        code = "import flagquantum as fq\nc = fq.Circuit(2).h(0).cx(0, 1)\nresult = fq.run(c)"
        if args.qsteed:
            code += '\ncompiled = fq.compile(c, compiler="qsteed")\nassert compiled.instructions'
        reply = client.execute_interactive(code, timeout=120)
        if reply["content"]["status"] != "ok":
            raise RuntimeError(f"Notebook kernel failed: {reply['content']}")
        print(f"Notebook kernel passed: {args.kernel}, qsteed={args.qsteed}")
    finally:
        client.stop_channels()
        manager.shutdown_kernel(now=True)


if __name__ == "__main__":
    main()
