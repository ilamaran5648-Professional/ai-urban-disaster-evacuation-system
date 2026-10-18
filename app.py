import json, heapq, math, os
from flask import Flask, jsonify, request, send_from_directory
from model import train, parts, level, FEATURES

ROOT = os.path.dirname(os.path.abspath(__file__))
# Works with folders (data/city.json, static/index.html) OR with all files in one flat folder
SD = os.path.join(ROOT, "static")
if not os.path.exists(os.path.join(SD, "index.html")): SD = ROOT
DP = os.path.join(ROOT, "data", "city.json")
if not os.path.exists(DP): DP = os.path.join(ROOT, "city.json")
app = Flask(__name__, static_folder=None)   # only index.html is served, never your .py files
D = json.load(open(DP))
MODEL = train()
N = {n["id"]: n for k in ("zones", "shelters", "hospitals") for n in D[k]}
SPEED_KMH = 20
ALERT_AT = 0.65      # a zone at or above this score raises an alert (High risk)

def km(a, b):
    (la1, lo1), (la2, lo2) = (a["lat"], a["lng"]), (b["lat"], b["lng"])
    p = math.pi/180
    h = math.sin((la2-la1)*p/2)**2 + math.cos(la1*p)*math.cos(la2*p)*math.sin((lo2-lo1)*p/2)**2
    return 12742*math.asin(math.sqrt(h))*1.25   # x1.25 = road winding factor

ROADS = [{"id": r["id"], "a": r["from"], "b": r["to"], "zone": r["zone"], "low": r["low_lying"],
          "status": r.get("status", "open"),
          "km": r.get("distance_km") or km(N[r["from"]], N[r["to"]])} for r in D["roads"]]

def dijkstra(src, adj):
    dist, prev, pq = {src: 0}, {}, [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]: continue
        for v, w, rid in adj.get(u, []):
            if d+w < dist.get(v, 1e18):
                dist[v], prev[v] = d+w, (u, rid); heapq.heappush(pq, (d+w, v))
    return dist, prev

def trace(prev, dst):
    path, rids = [dst], []
    while path[-1] in prev:
        u, rid = prev[path[-1]]; path.append(u); rids.append(rid)
    return path[::-1], rids[::-1]

def build(mult, thr, manual):
    # 1. AI risk score for every zone
    X = [[z["rain"]*mult, z["water"]*mult, z["elev"], z["pop"]] for z in D["zones"]]
    scores = MODEL.predict(X).clip(0, 1)
    Z = {}
    for z, x, s in zip(D["zones"], X, scores):
        s = float(s)
        why = {k: round(float(v), 3) for k, v in parts(*x).items()}
        Z[z["id"]] = {**z, "rain_now": round(x[0]), "water_now": round(x[1], 2), "score": round(s, 3),
                      "level": level(s), "why": why}

    # 2. Shelters are scored too: a flooded shelter is NOT a safe destination
    SX = [[s["rain"]*mult, s["water"]*mult, s["elev"], 500] for s in D["shelters"]]
    sscore = MODEL.predict(SX).clip(0, 1)
    shel = {}
    for s, sc in zip(D["shelters"], sscore):
        sc = float(sc)
        shel[s["id"]] = {**s, "used": 0, "score": round(sc, 3), "safe": sc < ALERT_AT}

    # 3. Blocked / unsafe roads, risk-weighted graph & physical distance graph
    adj, adj0, adj_open_km, roads = {}, {}, {}, []
    for r in ROADS:
        zs = Z[r["zone"]]["score"]
        reason = "manual closure" if r["id"] in manual else \
                 "reported blocked in input data" if r["status"] == "blocked" else \
                 f"flooded low-lying road in {Z[r['zone']]['name']}" if r["low"] and zs >= thr else ""
        roads.append({"id": r["id"], "coords": [[N[r["a"]]["lat"], N[r["a"]]["lng"]], [N[r["b"]]["lat"], N[r["b"]]["lng"]]],
                      "blocked": bool(reason), "reason": reason, "km": round(r["km"], 2)})
        adj0.setdefault(r["a"], []).append((r["b"], r["km"], r["id"])); adj0.setdefault(r["b"], []).append((r["a"], r["km"], r["id"]))
        if not reason:   # risk-weighted cost: safer roads are preferred even if slightly longer
            w = r["km"]*(1+2.5*zs)
            adj.setdefault(r["a"], []).append((r["b"], w, r["id"])); adj.setdefault(r["b"], []).append((r["a"], w, r["id"]))
            adj_open_km.setdefault(r["a"], []).append((r["b"], r["km"], r["id"])); adj_open_km.setdefault(r["b"], []).append((r["a"], r["km"], r["id"]))
    rl = {r["id"]: r["km"] for r in ROADS}

    # 4. Evacuation: worst zones first, nearest SAFE shelter with free places
    routes, notes = {}, []
    for s in shel.values():
        if not s["safe"]:
            notes.append(f"{s['name']} is flooded (score {s['score']}) and was removed from the shelter list")
    for z in sorted(Z.values(), key=lambda z: -z["score"]):
        need = 0 if z["level"] == "Low" else int(z["pop"]*(1 if z["level"] == "High" else .25))
        dist, prev = dijkstra(z["id"], adj)
        left, alloc = need, []
        for _, sid in sorted((dist[s], s) for s in shel if s in dist and shel[s]["safe"]):
            free = shel[sid]["cap"]-shel[sid]["used"]
            if left <= 0: break
            if free > 0:
                t = min(left, free); shel[sid]["used"] += t; left -= t; alloc.append({"shelter": sid, "people": t})
        z.update(evac=need, assigned=need-left, unplaced=left, alloc=alloc)
        hosp = sorted((dist[h["id"]], h["id"]) for h in D["hospitals"] if h["id"] in dist)
        z["hospital"] = None if not hosp else {"id": hosp[0][1], "km": round(sum(rl[r] for r in trace(prev, hosp[0][1])[1]), 2)}
        z["isolated"] = not hosp
        if alloc:
            dest = alloc[0]["shelter"]
            path_rec, rids_rec = trace(prev, dest)
            d0, p0 = dijkstra(z["id"], adj0)
            base = sum(rl[r] for r in trace(p0, dest)[1])
            k_rec = sum(rl[r] for r in rids_rec)
            rec_route = {"shelter": dest, "shelter_name": shel[dest]["name"], "roads": rids_rec, "km": round(k_rec, 2),
                         "eta_min": round(k_rec/SPEED_KMH*60), "detour_km": round(max(0, k_rec-base), 2),
                         "coords": [[N[n]["lat"], N[n]["lng"]] for n in path_rec], "via": [N[n]["name"] for n in path_rec]}

            # Alternative route: distinct valid route on open roads to safe shelter
            alt_route = None
            dist_km, prev_km = dijkstra(z["id"], adj_open_km)
            if dest in dist_km:
                path_alt, rids_alt = trace(prev_km, dest)
                if rids_alt != rids_rec:
                    k_alt = sum(rl[r] for r in rids_alt)
                    alt_route = {"shelter": dest, "shelter_name": shel[dest]["name"], "roads": rids_alt, "km": round(k_alt, 2),
                                 "eta_min": round(k_alt/SPEED_KMH*60), "type": "Shortest physical distance path",
                                 "coords": [[N[n]["lat"], N[n]["lng"]] for n in path_alt], "via": [N[n]["name"] for n in path_alt]}

            if not alt_route and len(rids_rec) > 0:
                bottleneck = max(rids_rec, key=lambda rid: rl[rid])
                adj_penalized = {}
                for u, nbrs in adj.items():
                    for v, w, rid in nbrs:
                        if rid != bottleneck:
                            adj_penalized.setdefault(u, []).append((v, w, rid))
                dist_pen, prev_pen = dijkstra(z["id"], adj_penalized)
                if dest in dist_pen:
                    path_alt, rids_alt = trace(prev_pen, dest)
                    if rids_alt != rids_rec:
                        k_alt = sum(rl[r] for r in rids_alt)
                        alt_route = {"shelter": dest, "shelter_name": shel[dest]["name"], "roads": rids_alt, "km": round(k_alt, 2),
                                     "eta_min": round(k_alt/SPEED_KMH*60), "type": f"Bypass path (bypassing {bottleneck})",
                                     "coords": [[N[n]["lat"], N[n]["lng"]] for n in path_alt], "via": [N[n]["name"] for n in path_alt]}

            if not alt_route:
                for sid in sorted((dist[s], s) for s in shel if s in dist and shel[s]["safe"] and s != dest):
                    s_id = sid[1]
                    path_alt, rids_alt = trace(prev, s_id)
                    if rids_alt and rids_alt != rids_rec:
                        k_alt = sum(rl[r] for r in rids_alt)
                        alt_route = {"shelter": s_id, "shelter_name": shel[s_id]["name"], "roads": rids_alt, "km": round(k_alt, 2),
                                     "eta_min": round(k_alt/SPEED_KMH*60), "type": f"Alternative safe shelter ({shel[s_id]['name']})",
                                     "coords": [[N[n]["lat"], N[n]["lng"]] for n in path_alt], "via": [N[n]["name"] for n in path_alt]}
                        break

            routes[z["id"]] = {**rec_route, "recommended": rec_route, "alternative": alt_route}
        elif need:
            notes.append(f"{z['name']}: no open path to any safe shelter")
        if z["unplaced"]:
            notes.append(f"{z['name']}: {z['unplaced']} people have no shelter place")

    open_cnt = sum(not r["blocked"] for r in roads)
    manual_cnt = sum(r["blocked"] and r["id"] in manual for r in roads)
    flood_cnt = sum(r["blocked"] and r["id"] not in manual for r in roads)

    alerts = [{"zone": z["name"], "id": z["id"], "score": z["score"]} for z in Z.values() if z["score"] >= ALERT_AT]
    zl = list(Z.values())
    fi = dict(zip(FEATURES, [round(float(v), 3) for v in MODEL.feature_importances_]))
    r2_val = getattr(MODEL, "holdout_r2", None)
    model_info = {
        "type": "Random Forest Regressor (Surrogate Model)",
        "data_source": "100% Synthetic Hydrology Simulation Data (6,000 samples)",
        "inputs": FEATURES,
        "risk_classes": {"Low": "< 0.35", "Medium": "0.35 - 0.65", "High": ">= 0.65"},
        "holdout_r2": r2_val,
        "feature_importance": fi,
        "limitation_disclaimer": "Model trained on synthetic rule-based simulation data for prototype evaluation. High holdout R² confirms surrogate fitting but does not establish real-world flood prediction accuracy. Production deployment requires training on historical sensor & telemetry datasets."
    }

    return {"zones": zl, "roads": roads, "routes": routes, "alerts": sorted(alerts, key=lambda a: -a["score"]),
            "shelters": [{**s, "safe_cap": s["cap"] if s["safe"] else 0, "free": (s["cap"]-s["used"]) if s["safe"] else 0} for s in shel.values()],
            "hospitals": D["hospitals"], "notes": notes, "alert_at": ALERT_AT,
            "summary": {"high": sum(z["level"] == "High" for z in zl), "medium": sum(z["level"] == "Medium" for z in zl),
                        "low": sum(z["level"] == "Low" for z in zl), "evacuees": sum(z["evac"] for z in zl),
                        "assigned": sum(z["assigned"] for z in zl), "unplaced": sum(z["unplaced"] for z in zl),
                        "total_cap": sum(s["cap"] for s in shel.values()),
                        "total_safe_cap": sum(s["cap"] for s in shel.values() if s["safe"]),
                        "free_beds": sum(s["cap"]-s["used"] for s in shel.values() if s["safe"]),
                        "open_roads": open_cnt, "manually_blocked": manual_cnt, "flood_blocked": flood_cnt,
                        "blocked": manual_cnt + flood_cnt,
                        "isolated": [z["name"] for z in zl if z["isolated"]]},
            "feature_importance": fi,
            "holdout_r2": r2_val,
            "model_info": model_info}

@app.get("/api/state")
def state():
    q = request.args
    manual = {b for b in q.get("blocked", "").split(",") if b}
    try:
        mult, thr = float(q.get("rain", 1)), float(q.get("thr", .7))
    except ValueError:
        return jsonify(error="rain and thr must be numbers"), 400
    return jsonify(build(mult, thr, manual))

@app.get("/")
def index():
    return send_from_directory(SD, "index.html")

if __name__ == "__main__":
    app.run(debug=os.environ.get("ARCOVA_DEBUG") == "1", port=5000)
