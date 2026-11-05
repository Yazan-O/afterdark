import argparse
import json
import sys

from .server import serve


def main():
    ap = argparse.ArgumentParser(prog="afterdark")
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--web", default=None)
    a = sub.add_parser("agent")
    a.add_argument("moment")
    a.add_argument("--route", choices=["spark", "glimmer"], default="spark")
    a.add_argument("--steps", type=int, default=14)
    a.add_argument("--held", type=int, default=None)
    audit_parser = sub.add_parser("audit")
    audit_parser.add_argument("generated")
    audit_parser.add_argument("events")
    args = ap.parse_args()
    if args.command == "serve":
        return serve(args.host, args.port, args.web)
    if args.command == "audit":
        from pipeline.audit import audit
        result = audit(args.generated, args.events)
        print(json.dumps(result))
        return 1 if result["errors"] else 0
    from .agent import AgentError, get_route, run_agent, write_trace
    from .moment import Moment
    try:
        route = get_route(args.route)
    except AgentError as e:
        sys.exit(str(e))
    moment = Moment(args.moment)

    def show(step):
        print(step["i"], step["tool"], step["seconds"], "s:", step["reason"], flush=True)

    trace = run_agent(moment, route, on_step=show, max_steps=args.steps, held=args.held)
    for path in write_trace(trace, moment, args.route):
        print("trace", path, flush=True)


sys.exit(main())
