#!/usr/bin/env python3
"""Telegram updates from the Verge run.

Sends a short population summary every --every minutes and an instant line for
the things worth knowing at once: a craft, a birth, an acceptance, a generation
turnover, or the swarm going quiet (state.json not updated for 2 min).

One-time setup (Telegram only lets bots talk to people who messaged them first):
  1. In Telegram, message @BotFather: /newbot -> it gives you a token like 123456:ABC...
  2. Put the token in results/verge/telegram.json:  {"token": "123456:ABC..."}
     (that file is in .gitignore; never commit it)
  3. Open your new bot in Telegram and send it any message (e.g. "hi").
  4. Run:  python3 verge_telegram.py --every 30
     The first run finds your chat from that "hi" and remembers it in telegram.json.
run_verge.ps1 starts this automatically when telegram.json exists.
"""
import argparse, json, os, time, urllib.request, urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "verge")
CFG = os.path.join(OUT, "telegram.json")


def api(token, method, **params):
    data = urllib.parse.urlencode(params).encode()
    with urllib.request.urlopen("https://api.telegram.org/bot%s/%s" % (token, method), data, timeout=20) as r:
        return json.load(r)


def find_chat(token):
    """The chat id of whoever messaged the bot last."""
    upd = api(token, "getUpdates")
    for u in reversed(upd.get("result", [])):
        m = u.get("message") or u.get("edited_message")
        if m and "chat" in m:
            return m["chat"]["id"], (m["chat"].get("username") or m["chat"].get("first_name") or "?")
    return None, None


def send(cfg, text):
    try:
        api(cfg["token"], "sendMessage", chat_id=cfg["chat_id"], text=text)
    except Exception as e:
        print("send failed:", e)


def load_state():
    try:
        with open(os.path.join(OUT, "state.json")) as f:
            return json.load(f)
    except Exception:
        return None


def summary(st):
    souls = st.get("souls", [])
    by = {}
    for s in souls:
        t = by.setdefault(s["tribe"], {"n": 0, "m": 0, "f": 0, "ate": 0, "hit": 0, "mem": [], "kids": 0, "made": set()})
        t["n"] += 1; t["m" if s["sex"] == "M" else "f"] += 1
        t["ate"] += s["life"]["forage"]; t["hit"] += s["life"]["hits"]; t["mem"].append(1 - s["weights"])
        t["kids"] += len(s.get("kids", []))
        for c in s.get("crafted_ever", []): t["made"].add(c["item"])
    lines = ["Verge, generation %d: %d alive, %d born so far" % (st.get("gen", 0), st.get("alive", 0), st.get("born", 0))]
    for k in sorted(by):
        t = by[k]
        lines.append("Tribe %d (%s): %d alive (%d♂ %d♀), ate %d, hit %d, kids %d, memory changed %.2f%%%s"
                     % (k, "inherits" if any(s["inherit"] for s in souls if s["tribe"] == k) else "naive",
                        t["n"], t["m"], t["f"], t["ate"], t["hit"], t["kids"], 100 * sum(t["mem"]) / max(len(t["mem"]), 1),
                        (", made " + ", ".join(sorted(t["made"]))) if t["made"] else ""))
    W = st.get("words", {})
    said = [(w, v["said"]) for w, v in W.items() if v.get("said")]
    if said:
        lines.append("Words: " + ", ".join("%s ×%d (came %d/left %d)" % (w, n, W[w]["approach"], W[w]["avoid"]) for w, n in said))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=float, default=30.0, help="minutes between summaries")
    a = ap.parse_args()
    if not os.path.exists(CFG):
        print("no %s - see the docstring for the three-step setup" % CFG); return
    cfg = json.load(open(CFG))
    if "chat_id" not in cfg:
        cid, who = find_chat(cfg["token"])
        if cid is None:
            print("send your bot a message in Telegram first, then run this again"); return
        cfg["chat_id"] = cid; json.dump(cfg, open(CFG, "w"))
        print("talking to", who)
    send(cfg, "Fly tribes in the Verge: updates on. Summary every %g min; crafts, births, acceptances and generation changes at once." % a.every)
    ev_path = os.path.join(OUT, "events.jsonl")
    pos = os.path.getsize(ev_path) if os.path.exists(ev_path) else 0      # only new events from now on
    last_sum = 0.0; quiet_sent = False
    while True:
        try:
            if os.path.exists(ev_path):
                with open(ev_path, encoding="utf-8") as f:
                    f.seek(pos)
                    for line in f:
                        try: e = json.loads(line)
                        except Exception: continue
                        k = e.get("event")
                        if k == "crafted": send(cfg, "⚒ Fly-%d-%02d made a %s (generation tick %s)" % (e["tribe"], e["soul"], e["item"], e.get("game_tick")))
                        elif k == "birth": send(cfg, "★ %s was born to soul %d and %s (%s)" % (e["child"], e["soul"], e.get("father"), "inherits" if e.get("inherits") else "naive"))
                        elif k == "mated": send(cfg, "♥ soul %d (Tribe-%d) accepted %s" % (e["soul"], e["tribe"], e["with"]))
                        elif k == "generation": send(cfg, "=== generation %d: eight new founders ===" % e["gen"])
                        elif k == "restart": send(cfg, "↻ restarted from the page")
                    pos = f.tell()
            st = load_state()
            if st:
                age = time.time() - st.get("wall", 0)
                if age > 120 and not quiet_sent:
                    send(cfg, "⚠ the swarm has been quiet for %d s (state.json not updating) - check results/verge/run.err" % age); quiet_sent = True
                elif age <= 120:
                    quiet_sent = False
                if time.time() - last_sum > a.every * 60:
                    last_sum = time.time(); send(cfg, summary(st))
        except Exception as ex:
            print("loop error:", ex)
        time.sleep(15)


if __name__ == "__main__":
    main()
