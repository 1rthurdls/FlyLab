"""Launch the FlyLab MCP server over stdio and call every tool (like Claude Code would).

Usage: python scripts/test_mcp.py [--quick]   (--quick skips run_behavior / lesion_screen)
"""
import asyncio
import json
import sys
import time
from pathlib import Path

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

REPO = Path(__file__).resolve().parents[1]
QUICK = "--quick" in sys.argv


async def main():
    params = StdioServerParameters(command=sys.executable, args=[str(REPO / "flylab" / "mcp_server.py")], cwd=str(REPO))
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = [t.name for t in (await s.list_tools()).tools]
            print("tools:", tools)
            calls = [
                ("search_neurons", {"query": "sugar", "limit": 5}),
                ("list_neuron_groups", {}),
                ("get_neuron_group", {"name": "MN9"}),
                ("trace_pathway", {"source": "sugar_GRNs", "target": "MN9"}),
                ("run_simulation", {"activate": ["sugar_GRNs"], "duration_s": 0.5}),
                ("run_simulation", {"activate": ["sugar_GRNs"], "silence": ["MN9"], "duration_s": 0.5}),
                ("get_literature_facts", {"topic": "MN9"}),
                ("list_behaviors", {}),
                ("save_report", {"title": "MCP test report", "markdown": "Test report from scripts/test_mcp.py."}),
            ]
            if not QUICK:
                calls += [("lesion_screen", {"activate": ["sugar_GRNs"], "target": "MN9", "n_candidates": 2,
                                             "duration_s": 0.5}),
                          ("run_behavior", {"behavior": "feed"}),
                          ("run_course", {"course": "canyon"}),
                          ("swat_fly", {"speed": "normal"}),
                          ("play_piano", {"mode": "orchestra"}),
                          ("run_maze", {}),
                          ("fly_through", {"place": "cave"})]
            results = {}
            ok = True
            for name, args in calls:
                t = time.perf_counter()
                res = await s.call_tool(name, args)
                text = res.content[0].text
                data = json.loads(text)
                err = isinstance(data, dict) and "error" in data
                ok &= not err
                results[name] = data
                print(f"{'FAIL' if err else 'ok  '} {name:22s} {time.perf_counter() - t:6.1f}s  {text[:150].replace(chr(10), ' ')}")
            a = results["run_simulation"]
            # compare control vs lesion (the two run_simulation calls above)
            res = await s.call_tool("compare_conditions", {"run_id_a": "sim_sugar-GRNs_" + "x", "run_id_b": "x"})
            print("compare_conditions with bad ids ->", res.content[0].text[:80])
            assert set(tools) >= {c[0] for c in calls} | {"compare_conditions", "ride_bike", "run_olympics"}
            print("ALL OK" if ok else "SOME TOOLS FAILED")
            return ok


if __name__ == "__main__":
    sys.exit(0 if asyncio.run(main()) else 1)
